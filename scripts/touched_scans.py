"""The source-scan tests touched.py must run when a file they read changes.

A source-scan test reads the repository's own files (Go source, kernel source) and checks a property across them: a census
of dispatch sites, a switch's case labels, every route's guard. Selecting tests by the test file beside a changed file misses
it, because the test lives elsewhere, often in another package or module. SCANS lists each one with the paths it reads;
select() maps a set of changed paths to the tests to run. scripts/test_touched.py fails on a test that reads Go source and is
in neither SCANS nor EXCLUDED, so a new scanner cannot be missed silently.

SCANS also lists the censuses and gates that read no source but pin, from one package, what another declares (the cache's
fields, the family registry, a generated table): the package they live in is not the one that changes. problems() names every
entry the tree has moved away from, and touched.py fails on one, so a moved file cannot leave its scan unselected.
"""
import fnmatch
import os
import re
import subprocess

SKIP_DIRS = ("testdata/", "node_modules/", ".git/", "apps/site/")


def F(*paths):
    """These exact files."""
    return [("file", p) for p in paths]


def NT(*dirs):
    """Every non-test .go file directly in each dir."""
    return [("nt", d) for d in dirs]


def T(*dirs):
    """Every _test.go file directly in each dir."""
    return [("t", d) for d in dirs]


def CU(*dirs):
    """Every .cu file directly in each dir."""
    return [("cu", d) for d in dirs]


def UNDER(*dirs):
    """Every non-test .go file under each dir, subdirectories included."""
    return [("under_nt", d) for d in dirs]


def PATHS_OF(*workflows):
    """Every path a pull request must change for the workflow to run: its pull_request trigger's paths list."""
    return [("workflow_paths", w) for w in workflows]


def NAMED_IN(*patterns):
    """Every path the text of a file names, for the files a pattern (dir/glob) matches: a path one of them spells out, and each
    file that was under a directory one of them spells out, once that directory has left the tree."""
    return [("named_in", p) for p in patterns]


WHOLE = "*"  # in place of a test name: the package's whole test run


TREE_NT = [("tree_nt", "")]  # every non-test .go file in the repository, outside SKIP_DIRS
TREE_T = [("tree_t", "")]    # every _test.go file in the repository, outside SKIP_DIRS
TREE_GO = TREE_NT + TREE_T


class Scan:
    def __init__(self, pkg, test, scopes):
        self.pkg, self.test, self.scopes = pkg, test, scopes


def _matches(kind, arg, path):
    d, base = os.path.split(path)
    if kind == "file":
        return fnmatch.fnmatch(path, arg)
    if kind == "nt":
        return d == arg and base.endswith(".go") and not base.endswith("_test.go")
    if kind == "t":
        return d == arg and base.endswith("_test.go")
    if kind == "cu":
        return d == arg and base.endswith(".cu")
    skipped = any(path.startswith(s) or ("/" + s) in path for s in SKIP_DIRS)
    if kind == "under_nt":
        return not skipped and path.startswith(arg + "/") and path.endswith(".go") and not path.endswith("_test.go")
    if kind == "tree_nt":
        return not skipped and path.endswith(".go") and not path.endswith("_test.go")
    if kind == "tree_t":
        return not skipped and path.endswith("_test.go")
    raise ValueError(kind)


def _workflow_paths(root, workflow):
    """The pull_request trigger's paths list of a workflow file under root, in order, each entry as (negated, compiled pattern);
    empty when the file is gone or has no such list. Reads the block form (a "- pattern" line each) and GitHub's ** (any
    characters) and * (any but /); a leading ! negates."""
    try:
        with open(os.path.join(root, workflow), encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    out, keys = [], []  # keys: the mapping keys enclosing the current line, each as (indent, key)
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        item = re.fullmatch(r"-\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))\s*(?:#.*)?", text)
        if item and [k for _, k in keys] == ["on", "pull_request", "paths"]:
            pat = next(g for g in item.groups() if g is not None)
            neg = pat.startswith("!")
            rx = "".join(".*" if t == "**" else "[^/]*" if t == "*" else re.escape(t) for t in re.split(r"(\*\*|\*)", pat[neg:]))
            out.append((neg, re.compile(rx)))
            continue
        key = re.match(r"(['\"]?)([\w.-]+)\1\s*:", text)
        if key:
            indent = len(line) - len(line.lstrip(" "))
            while keys and keys[-1][0] >= indent:
                keys.pop()
            keys.append((indent, key.group(2)))
    return out


def _path_filter(patterns, path):
    """Whether a workflow's paths list (_workflow_paths) selects path: the last pattern that matches it decides."""
    hit = False
    for neg, rx in patterns:
        if rx.fullmatch(path):
            hit = not neg
    return hit


def _named_paths(root, pattern):
    """The paths spelled out in the files directly in the pattern's directory under root whose name matches its glob: each run
    of path characters, and each tail of one after a / (a link names a path after its host and branch). A longer path that
    only begins with a path does not name it."""
    d, glob = os.path.split(pattern)
    try:
        bases = sorted(b for b in os.listdir(os.path.join(root, d)) if fnmatch.fnmatch(b, glob))
    except OSError:
        return frozenset()
    out = set()
    for b in bases:
        with open(os.path.join(root, d, b), encoding="utf-8", errors="replace") as fh:
            for run in re.findall(r"[\w./-]+", fh.read()):
                run = run.rstrip("./")  # the full stop that ends a sentence, a directory's trailing slash
                out.add(run)
                cut = run.find("/")
                while cut >= 0:
                    out.add(run[cut + 1:])
                    cut = run.find("/", cut + 1)
    return frozenset(out)


def _reads(kind, arg, path, root, memo):
    """_matches for every kind of scope. The two that read the tree at root to decide are answered here, from memo once read."""
    if kind not in ("workflow_paths", "named_in"):
        return _matches(kind, arg, path)
    if (kind, arg) not in memo:
        memo[kind, arg] = _workflow_paths(root, arg) if kind == "workflow_paths" else _named_paths(root, arg)
    if kind == "workflow_paths":
        return _path_filter(memo[kind, arg], path)
    named = memo[kind, arg]
    if path in named:
        return True
    # A directory is named by its own path, and a file under it breaks nothing while the directory stands. Once the directory
    # is gone, each of its files is in the diff as a deletion.
    d = os.path.dirname(path)
    while d and not os.path.lexists(os.path.join(root, d)):
        if d in named:
            return True
        d = os.path.dirname(d)
    return False


# decoder and the runtime packages split out of it: where the families, the forward, the cache and their pricing are declared.
RUNTIME_NT = UNDER("decoder", "internal/runtime")

SCANS = [
    # decoder
    Scan("decoder", "TestDispatchCensus", NT("decoder", "internal/runtime/cpuengine")),
    Scan("decoder", "TestKVRowPad_keysReadersAreDeclinedFamilies", NT("decoder")),
    Scan("decoder", "TestResidentSubsystems_NoScatteredMicroAssertions", NT("decoder", "internal/runtime/residency")),
    # reads every .go file under each backend module, subdirectories included (fnmatch's * crosses /)
    Scan("decoder", "TestResidentSubsystems_NoScatteredMicroAssertionsInBackends", F("backend/cuda/*.go", "backend/metal/*.go", "backend/webgpu/*.go")),
    # MM1's decision census pins every fit decision: a change to the fit code, the machine zoo or a fixture it reads runs it. Every
    # censusFixtures entry has "tiny" in its name (TestCensusFixtures_areTheCommittedOnes).
    Scan("decoder", "TestStepShapes_assertedOnlyInStep", NT("decoder")),
    Scan("decoder", "TestDecisionCensus", F("decoder/fitguard.go", "decoder/fitplan.go", "decoder/fitquery.go", "decoder/prefill_budget.go",
                                            "decoder/weightbytes.go", "decoder/residentneed.go", "decoder/kvresolve.go", "decoder/kvpolicy.go",
                                            "decoder/metal_fitctx.go", "decoder/memprobe.go", "testdata/machines/*.json", "testdata/census/*.txt.gz",
                                            "decoder/mm1_census_fixtures_test.go", "testdata/*tiny*", "decoder/testdata/*tiny*",
                                            "decoder/selftestdata/*tiny*", "internal/vision/testdata/*tiny*",
                                            # MM5: KV and state are priced from each layer's operations' declarations
                                            "decoder/statebytes.go", "decoder/membufs.go", "decoder/program.go",
                                            "internal/runtime/program/buffers.go") +
         NT("internal/runtime/mem", "internal/runtime/memprice")),
    # docs/option-state-grid.md is generated from the cache-state table's text (decoder/cachestate.go), and the test also checks the function and test names it cites: any non-test decoder file can move it.
    Scan("decoder", "TestOptionStateGrid_fresh", NT("decoder") + F("decoder/optiongrid_test.go", "docs/option-state-grid.md")),
    # MM5's G-MP4: KV and state pricing names no family or mixer.
    Scan("decoder", "TestMM5_noFamilySwitchInPricing", F("decoder/residentneed.go", "decoder/statebytes.go", "decoder/arch.go", "decoder/fitplan.go") + NT("internal/runtime/memprice")),
    # MM5b's G-MS4: Metal's decode scratch pricing names no family, mixer or FFN form.
    Scan("decoder", "TestMM5b_decodeScratchNamesNoFamily", F("decoder/metal_scratch.go", "internal/runtime/memprice/metal.go")),
    Scan("internal/runtime/memprice", "TestNoFamilySwitchInPricing", NT("internal/runtime/memprice")),
    Scan("internal/runtime/memprice", "TestDecodeScratchNamesNoFamily", F("internal/runtime/memprice/metal.go")),
    Scan("internal/runtime/wbuild", "TestWbuild_everyTensorReadReleases", NT("internal/runtime/wbuild")),
    # The census reads a declared fixture list, which must match the committed fixtures: a fixture added or removed runs it.
    Scan("decoder", "TestCensusFixtures_areTheCommittedOnes", F("decoder/mm1_census_fixtures_test.go", "testdata/*", "decoder/testdata/*",
                                                                "decoder/selftestdata/*", "internal/vision/testdata/*")),
    # VK2's census: each family's own list covers its registry adapter's file and the arch_*.go files it reaches.
    Scan("decoder", "TestParityManifest_adapterFilesOwned", NT("decoder", "internal/runtime/families") + F("testdata/parity/*.json")),
    # VK2's forward census: every file the forward reaches is in a hash set or an own list.
    Scan("decoder", "TestParityManifest_coreClosed", NT("decoder", "internal/runtime/cpuengine") + F("testdata/parity/*.json")),
    Scan("decoder", "TestParityManifest_loadersClosed", NT("decoder", "decoder/quant", "decoder/config", "internal/runtime/families", "internal/runtime/wbuild") + F("testdata/parity/*.json")),
    # A record that hashes a test file running through textParity hashes the helper too.
    Scan("decoder", "TestTextParity_hashedWithItsUsers", T("decoder") + F("testdata/parity/*.json")),
    Scan("decoder", "TestTranscode_writesViaTempThenRenames", F("decoder/sidecar.go", "internal/runtime/sidecar/sidecar.go")),
    Scan("decoder", "TestTranscodeDir_writesViaTempThenRenames", F("decoder/sidecar.go", "internal/runtime/sidecar/sidecar.go")),
    Scan("decoder", "TestStreamableFamilyClosures_onlyReadPerLayerTensors", F("decoder/gguf.go", "internal/runtime/wbuild/gguf.go", "internal/runtime/wbuild/gguf_build*.go")),
    Scan("decoder", "TestEmbedInt4_noLoaderPinsTheHeadItself", NT("decoder", "internal/runtime/wbuild")),
    Scan("decoder", "TestProgram_noFamilyLoopAndEveryFamilyWalks", NT("decoder", "internal/runtime/cpuengine")),
    Scan("decoder", "TestFP8_shapeCheckDistinguishesTranspose", F("decoder/quant/fp8.go")),
    Scan("decoder", "TestGoDoc_listsEveryFieldOfOptionsAndSamplingParams", NT("decoder")),
    Scan("decoder", "TestBackendBanner_usesTheReport", TREE_NT),
    Scan("decoder", "TestEnvVars_docAndCodeAgree", TREE_GO + F("docs/env-vars.md", "testdata/env_reads.txt")),
    Scan("decoder", "TestEnvVars_noNewPackageInitRead", TREE_NT),
    Scan("internal/config", "TestInventory", F("decoder/knobs.go", "decoder/model.go", "testdata/env_reads.txt", "docs/flags.meta.json") + NT("internal/app/loadflags")),
    Scan("internal/config", "TestInventory_plants", F("decoder/knobs.go", "decoder/model.go", "testdata/env_reads.txt", "docs/flags.meta.json") + NT("internal/app/loadflags")),
    Scan("internal/config", "TestEnvVarsDoc", F("docs/env-vars.md", "docs/env-vars.meta.json")),
    Scan("internal/config", "TestEnvVarsDoc_plants", F("docs/env-vars.meta.json")),
    Scan("internal/app/serveapp", "TestOneMerge", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestFlagsDoc", F("docs/flags.md", "docs/flags.meta.json") + NT("internal/app/serveapp")),
    Scan("decoder", "TestGoldenNames_matchTheFileOnDisk", TREE_T),
    Scan("decoder", "TestOptionGrid_cellsCarryEvidence", TREE_GO),
    Scan("internal/testassets", "TestAssetRegistry_usedByExists", TREE_T + F("testdata/assets.json", "internal/testassets/assets.go")),
    Scan("internal/testassets", "TestAssetRegistry_noDirectReads", TREE_T + F("testdata/assets.json", "internal/testassets/assets.go")),
    Scan("internal/testassets", "TestAssetRegistry_resolvesFromEveryCallerDir", TREE_T + F("testdata/assets.json", "internal/testassets/assets.go")),
    # Two computations of one key over the same files: only the two implementations or the manifest can split them.
    Scan("decoder", "TestParityKey_matchesFreshDepsHash", F("decoder/parity_manifest_test.go", "testdata/parity/*.json", "testdata/kernel_fingerprints.json", "testdata/parity_key.json", "testdata/program_fingerprints.json") + NT("internal/guard/paritykey")),
    Scan("decoder", "TestParityKey_shadowMatchesDecoder", F("testdata/loader_identity.json", "decoder/parity_manifest_test.go", "testdata/parity/*.json", "testdata/kernel_fingerprints.json", "testdata/program_fingerprints.json", "testdata/parity_key.json") + NT("internal/guard/paritykey", "decoder")),
    # Both keys of every family over the working tree, from the tables they read.
    Scan("internal/guard/paritykey", "TestKey_onTheTree", F("testdata/parity/*.json", "testdata/parity_key.json", "testdata/program_fingerprints.json", "testdata/loader_identity.json", "testdata/kernel_fingerprints.json")),
    Scan("decoder", "TestParityPrograms_mapping", F("testdata/parity/*.json", "testdata/program_fingerprints.json", "testdata/*/config.json") + NT("decoder", "internal/runtime/families")),
    # decoder's config tests exercise decoder/config through decoder's wrappers (loadConfig, resolveEOSIDs) and real fixtures, so an
    # edit in decoder/config runs them; touched.py alone would run only decoder/config's own tests (where Config's validator tests live).
    *[Scan("decoder", n, NT("decoder/config")) for n in (
        "TestValidateConfigBounds_hostileCountsAreRefusedBeforeAllocation", "FuzzLoadConfig",
        "TestConfigSnapshot_M23", "TestLagunaArchitecture_realConfigs", "TestMellum21_configResolves", "TestNemotronH_attentionHeadDim",
        "TestRecommendedSampling_sources", "TestQwen3ASR_loadedModelDoesNotStopOnIDZero")],
    # The parity key's and the program fingerprints' guards: Config and the program's types, methods and constants are builders or
    # core files of every family, in packages whose own tests touched.py would run alone.
    *[Scan("decoder", n, NT("decoder/config", "internal/runtime/program")) for n in (
        "TestParityKey_matchesFreshDepsHash", "TestParityKey_shadowMatchesDecoder", "TestProgramFingerprint_table", "TestParityPrograms_mapping")],
    Scan("decoder", "TestParityLoaders_mapping", F("testdata/parity/*.json", "testdata/loader_identity.json") + NT("internal/guard/paritykey")),
    # It polices one file of the program package, and a declaration can move into it from any other.
    Scan("internal/guard/paritykey", "TestBuilderFile_programTypesDeclareOnly", NT("internal/runtime/program")),
    # The fixture censuses walk testdata/: a new or changed model fixture must be listed or excluded in each (8ec704c3).
    Scan("decoder", "TestSerializeCensus_everyFixtureIsListedOrExcluded", F("testdata/*/config.json")),
    Scan("decoder", "TestSeams_subCaptureRefusedWhereGateSCannotRun", F("testdata/*/config.json")),
    Scan("decoder", "TestParityEmit_methodVocabulary", TREE_T),
    Scan("decoder", "TestResidentMoECap_pinsKernelConstant", F("backend/webgpu/internal/kernels/moe.go", "backend/cuda/moe.cu")),
    Scan("decoder", "TestResidentMoECap_runtimeDeclineReadsTheDeclaration", F("backend/webgpu/residency.go")),
    # Censuses and generated tables over the family registry and the cache's declarations. Each compares what the code declares
    # with a table, a golden or a generated doc and reads no Go source, so the reader check in scripts/test_touched.py does not
    # find it; decoder is a slow package, so touched.py runs them only from here.
    *[Scan("decoder", n, RUNTIME_NT + F(*also)) for n, also in (
        ("TestKVCache_everyFieldHasAState", ()),
        ("TestKVCache_recurrentKindsResetAndRewindHonestly", ()),
        ("TestResidentAdmission_registryCovered", ()),
        ("TestResidentAdmission_matrix", ()),
        ("TestHardwareMatrix_fresh", ("docs/hardware-matrix.md", "docs/hardware-coverage.json")),
        ("TestHardwareMatrix_declineReasonsNotHidden", ()),
        ("TestVerifiedOn_followsTheCensus", ("docs/hardware-coverage.json",)),
        ("TestKVBytesForCtx_isTheDeclaredRows", ()),
        ("TestKVBytesForRows_plantedShortRowGoesRed", ()),
        ("TestWebGPUKV_isTheDeclaredRows", ()),
        ("TestKVPrecisionMatrix_fresh", ("docs/kv-precision-matrix.md",)),
        ("TestLoaderIdentity_table", ("testdata/loader_identity.json",)),
        ("TestCapabilityMatrix", ("docs/capability-matrix.json", "docs/capability-matrix.md", "docs/hardware-coverage.json",
                                  "pull/capability-matrix.json")),
        ("TestCapabilityMatrix_checkpointLabels", ()),
        ("TestCapabilityMatrix_siteDocsAreUsable", ()),
        ("TestCapabilityMatrix_CoverageComplete", ()),
        ("TestSupportTable_fresh", ("README.md", "docs/multimodal.md")),
        ("TestSerializeCensus_noSilentFieldDrop", ()),
        ("TestGIWCopiedF32_everyFamily", ()),
        ("TestEmbedInt4_reachesEveryFamilyLoader", ()),
        ("TestExpertPaging_touchesEveryFamilyThatBuildsAPager", ()),
    )],
    # MM7's CPU gate loads every census fixture and holds the plan's KV and state lines to what a forward leaves allocated.
    *[Scan("internal/guard/memgate", n, RUNTIME_NT + F("testdata/*/config.json", "testdata/sb2_admission_census.*.txt.gz"))
      for n in ("TestMemGate_cpu", "TestMemGate_cpuPlantsThroughTheTable", "TestMemGate_cpuRingsAcrossTheWindow")],
    # Every parity-manifest family has an identity asset or a recorded reason for none.
    Scan("internal/guard/gateidentity", "TestIdentityAssets_coverTheManifest", F("testdata/parity/*.json")),
    # The README's family count is the generated capability matrix's length.
    Scan("pull", "TestReadme_familyCountMatchesTheMatrix", F("README.md", "docs/capability-matrix.json")),
    # The ratchets the lint lane holds every pull request to (function and file length, decoder's ceilings, forwarding shims and
    # the rest): each parses Go files from disk, in every module, and its allowlist may only shrink.
    Scan("internal/guard/qualityscan", WHOLE, TREE_GO + F("internal/guard/qualityscan/testdata/*")),
    # internal/app/serveapp
    Scan("internal/app/serveapp", "TestBanner_contextIsTheEnforcedWindow", F("internal/app/serveapp/banner.go", "internal/app/serveapp/model_load.go")),
    Scan("internal/app/serveapp", "TestServeVisionChat_appliesExtractionRuleBeforeThePromptIsBuilt", F("internal/app/serveapp/vision_serve.go")),
    Scan("internal/app/serveapp", "TestServe_everyTokenizingRouteGuardsItsInputSize", NT("internal/app/serveapp")),
    # A loadedModel's blockSpec is set only through drafterOf.
    Scan("engine", "TestSurface_exportsOnlyItsContract", NT("engine", "engine/enginetest")),
    Scan("internal/app/serveapp", "TestBlockDrafter_assignedThroughDrafterOf", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp/sessions", "TestSessions_persistedStateIsOwnerOnly", F("internal/app/serveapp/sessions/sessions.go")),
    Scan("internal/app/serveapp", "TestAdminLoad_racedDuplicateIsClosed", F("internal/app/serveapp/admin.go", "internal/app/serveapp/webui.go")),
    Scan("internal/app/serveapp", "TestStartDrain_savesSessionsUnderTheLRULock", F("internal/app/serveapp/liveness.go")),
    Scan("internal/app/serveapp", "TestPanicGuard_everyGoFuncLiteralDefersARecover", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestAdminLoad_requestQuantIsTheExplicitOne", F("internal/app/serveapp/admin.go")),
    Scan("internal/app/serveapp", "TestLookupLockedOnlyViaResolve", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestStreamSurfaces_allSendUsageBeforeDone",
         F("internal/app/serveapp/openai.go", "internal/app/serveapp/tools.go", "internal/app/serveapp/vision_serve.go")),
    Scan("internal/app/serveapp", "TestStreamTokens_decodesAsAContinuation", F("internal/app/serveapp/openai_generate.go")),
    Scan("internal/app/serveapp", "TestServe_prepareEnforcesThePublishedWindow",
         F("internal/app/serveapp/openai_generate.go", "internal/app/serveapp/openai.go")),
    Scan("internal/app/serveapp", "TestDriveUsesRequestContext", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestRespondTools_callsMaybeStoreWithTheToolCalls", F("internal/app/serveapp/responses.go")),
    Scan("internal/app/serveapp", "TestSSE_everyStreamingDriveSiteHeartbeats",
         F("internal/app/serveapp/openai.go", "internal/app/serveapp/anthropic_stream.go", "internal/app/serveapp/vision_serve.go",
           "internal/app/serveapp/responses.go")),
    Scan("internal/app/serveapp", "TestSSE_everyBufferedStreamSiteHeartbeats", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestSupportTable_everyFamilyHasALoader", F("internal/app/serveapp/towers_load.go", "docs/multimodal.md")),
    Scan("internal/app/serveapp", "TestServe_noRouteEncodesARenderedPrompt", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestVision_imageBlockNewlinesMatchTheRealProcessors",
         F("internal/app/serveapp/towers/vision_tower.go", "internal/app/serveapp/vision_multi.go", "internal/app/serveapp/vision_serve.go")),
    Scan("internal/app/serveapp", "TestWebUI_rootRouteIsUnauthenticated", F("internal/app/serveapp/main.go")),
    Scan("internal/app/serveapp", "TestWebUI_pageRoutesExistOnlyUnderWebFlag", F("internal/app/serveapp/main.go")),
    Scan("internal/app/serveapp", "TestWebUI_listAndPullAreWrappedInSameOrigin", F("internal/app/serveapp/main.go")),
    # other root-module packages
    Scan("internal/app/cliutil", "TestLdflagsTargetsExist", NT("internal/app/chatapp", "internal/app/serveapp")),
    Scan("internal/app/fitcmd", "TestDefaultCtx_agreesWithTheCudaPlannerAndHoldsAnAgentTurn", F("backend/cuda/resident.go", "internal/app/fitcmd/fit.go")),
    Scan("internal/app/fitcmd", "TestRun_closesFirstLoadBeforeMeasuring", F("internal/app/fitcmd/fit.go")),
    Scan("internal/diag/hwcensus", "TestHardwareCensus", TREE_NT + F("docs/hardware-coverage.json")),
    Scan("internal/runtime/mem/probe", "TestOnlyProbeReadsOSMemory", TREE_NT),
    Scan("internal/guard/testcmdtags", "TestDocs_goTestCommandsCompileTheTestsTheyName", TREE_T + F("docs/*.md")),
    Scan("internal/testpath", "TestNoStrayRelativePathInInternalTests", TREE_T),
    Scan("internal/runtime/mem", "TestImports", NT("internal/runtime/mem")),
    Scan("backend/cuda", "TestPlanSlack_oneCopy", NT("backend/cuda")),
    Scan("backend/cuda", "TestLedger_everyAllocationNamesALine", NT("backend/cuda")),
    Scan("internal/layout", "TestCUDASoloShapes_takeTodaysCalls", F("backend/cuda/resident.go", "backend/cuda/gumbel.go")),
    Scan("internal/runtime/mem", "TestSolve_pure", NT("internal/runtime/mem")),
    Scan("internal/runtime/mem", "TestNoGuardThresholdOutsideMem", TREE_NT),
    Scan("internal/runtime/mem", "TestRecover_everyBroadRecoverLetsABugThrough", TREE_NT),
    Scan("internal/runtime/mem/probe", "TestProbeImports", NT("internal/runtime/mem/probe")),
    # internal/testgate is imported by _test.go files only, in every module (its env reads are test-only).
    Scan("internal/testgate", "TestNoProductionImporter", TREE_GO),
    Scan("internal/layout", "TestBackendTests_decoderImportsOnlyFall", TREE_GO),
    # SB7: internal/app/serveapp test files importing decoder may only fall.
    Scan("internal/layout", "TestServeappTests_decoderImportsOnlyFall", TREE_GO),
    # Mutable package-global *ForTest variables in non-test files may only fall.
    Scan("internal/layout", "TestForTestGlobals_onlyFall", TREE_NT),
    # SB13: every goinfer-to-goinfer import against internal/layout/imports.json, and the root go.mod against its rules.
    # Every fuzz target is in the weekly matrix, under the directory it lives in now.
    Scan("internal/layout", "TestFuzzWeekly_listsEveryTarget", TREE_T + F(".github/workflows/fuzz-weekly.yml")),
    Scan("internal/layout", "TestLayout_arrows", TREE_GO + F("internal/layout/imports.json", "go.mod")),
    # SB13: towers-standalone's one exception (internal/audio may import internal/vision), read from imports.json.
    Scan("internal/layout", "TestLayout_towersStandalone", F("internal/layout/imports.json")),
    # SB14: every package doc's named tests exist; docs/ARCHITECTURE.md's import appendix is imports.json, rendered.
    Scan("internal/layout", "TestPackageDocs_namedTestsExist", TREE_GO),
    Scan("internal/layout", "TestArchitecture_importAppendix", F("internal/layout/imports.json", "docs/ARCHITECTURE.md")),
    Scan("internal/guard/prezerolint", "TestPreZero_noDeadZeroingBeforeOverwriteKernels", TREE_NT),
    # Library code writes through internal/runtime/logs, and no block both reports an error and returns it.
    Scan("internal/runtime/logs", "TestLibraryOutput_goesThroughLogs", TREE_NT),
    Scan("internal/guard/movedpaths", "TestNoMovedAikitPath", TREE_GO + F("go.mod", "*/go.mod", "go.sum", "*/go.sum")),
    Scan("pull", "TestResolve_callsResolveOfflineBeforeCheckAccess", F("pull/resolve.go")),
    Scan("apps/examples/structured", "TestUseFromGoDoc_codeIsFromExamples", F("apps/examples/*/main.go", "docs/use-from-go.md")),
    Scan("internal/guard/gatehygiene", "TestServeRoutes_findsTheRealOnes", NT("internal/app/serveapp")),
    Scan("decoder", "TestPolicyTests_loadedRatchet", F("decoder/spec_*_test.go", "decoder/speculative*_test.go", "decoder/session*_test.go", "testdata/sb5/loaded_policy_tests.txt")),
    Scan("decoder", "TestPolicyStreams_census", F("testdata/sb5/policy_streams.json")),
    Scan("cmd/gate", "TestHeavyPartition_cudaTreeFitsTheTable", T("backend/cuda")),
    Scan("cmd/gate", "TestGPU_isolatedGroupIsDerivedFromTheMarker", T("backend/cuda")),
    Scan("internal/guard/gatequick", "TestQuick_slowPatternsNameRealTests", T("backend/webgpu", "backend/cuda", "backend/metal")),
    Scan("internal/guard/gatequick", "TestQuick_checkpointAloneNamesExist", T("decoder")),
    Scan("cmd/gate", "TestWebGPUGateIsListedOrExplicitlyNotRequired", T("backend/webgpu")),
    Scan("cmd/gate", "TestRealckptGateIsListedOrExplicitlyNotRequired", T("decoder")),
    Scan("cmd/gate", "TestRealckptRunReachesEveryScannedAndRequiredGate", T("decoder")),
    Scan("cmd/gate", "TestRealckptCellCanReachEveryGate", T("decoder")),
    Scan("cmd/gate", "TestGateRunFilter_narrowsTheRealckptPattern", T("decoder")),
    Scan("cmd/gate", "TestGraderCensus_everySubcommand", F("cmd/gate/main.go", "docs/measurements/sb12-2026-10.md") + NT("cmd/gate", "internal/guard/gatecore", "internal/guard/gatehygiene", "internal/guard/gateidentity", "internal/guard/gatequick", "internal/guard/gaterelease")),
    Scan("cmd/gate", "TestGraders_arePure", NT("cmd/gate", "internal/guard/gatecore", "internal/guard/gatehygiene", "internal/guard/gateidentity", "internal/guard/gatequick", "internal/guard/gaterelease")),
    Scan("cmd/gate", "TestGraderGoldens", F("testdata/gate/goldens/*", "docs/measurements/giw-v16-transcode-2026-10/*")),
    Scan("cmd/gate", "TestRealckptRunIsAdditiveOverTheLegacyPattern", T("decoder")),
    Scan("cmd/gate", "TestMetalGateIsListedOrExplicitlyNotRequired", T("backend/metal")),
    # demo/agent module
    Scan("demo/agent/agent", "TestAgentWeb_hardening", F("demo/agent/cmd/agent-web/main.go")),
    Scan("demo/agent/agent", "TestAgentSession_encodesSegmentsNotRenderedText", F("demo/agent/agent/agent.go")),
    Scan("demo/agent/agent", "TestTurnImage_wiresSpliceImageBlockBeforeEncoding", F("demo/agent/agent/agent.go")),
    # site module
    Scan("apps/site/internal/site", "TestHome_heroCodeIsFromTheExample", F("apps/examples/structured/main.go")),
    # The site's tests build the site from the repository: a page lists the files it rests on, links into the tree and names
    # the record each of its figures comes from, and the build refuses a page whose file is gone or whose record no longer
    # holds the figure. They run for a path site.yml runs for, and for a path a page or a data file names.
    Scan("apps/site/internal/site", WHOLE, PATHS_OF(".github/workflows/site.yml")
         + NAMED_IN("apps/site/different/*.md", "apps/site/data/*.json")),
    # gpu module
    # MM5b's G-MS4w: WebGPU's decode scratch pricing names no family or mixer.
    Scan("backend/webgpu", "TestMM5b_webgpuDecodeScratchNamesNoFamily", F("backend/webgpu/sb2_census_test.go")),
    Scan("backend/webgpu", "TestNemotronBlockKind_hasADefault", F("backend/webgpu/residency.go", "backend/webgpu/residency_layer.go")),
    Scan("backend/webgpu/internal/kernels", "TestKernels_noDroppedTryError", NT("backend/webgpu/internal/kernels")),
    Scan("backend/webgpu/internal/kernels", "TestLedger_everyBufferIsLeased", NT("backend/webgpu", "backend/webgpu/internal/kernels")),
    Scan("backend/webgpu/internal/kernels", "TestRopeKernels_noBuiltinSinCos", NT("backend/webgpu/internal/kernels")),
    Scan("backend/webgpu", "TestEnsureAttnWide_guardsOnTheLastPipeline", F("backend/webgpu/internal/kernels/attention.go")),
    Scan("backend/webgpu", "TestGPU_deviceBoundaryErrorsAreNotDropped", F("backend/webgpu/internal/kernels/decoderunner.go", "backend/webgpu/residency.go")),
    Scan("backend/webgpu", "TestResidency_kvPrecisionDeclineCoversTheNonGenericBranches", F("backend/webgpu/residency.go", "backend/webgpu/residency_build.go")),
    # metal module
    Scan("backend/metal", "TestAttnFABlkIsTheGradedKernel", F("backend/metal/kernels/decode_attention_fa_blk.metal")),
    Scan("backend/metal", "TestGptOssBiasDispatches_alwaysIndexedByRealExpert", F("backend/metal/moe.go")),
    Scan("backend/metal", "TestLedger_everyMetalAllocationNamesALine", NT("backend/metal", "backend/metal/internal/gpudev")),
    Scan("backend/metal", "TestSwitchGuard_coversEveryAssignedSwitch", NT("backend/metal") + T("backend/metal")),
    Scan("backend/metal/internal/gpudev", "TestOutputContract_newBufferDocsAgreeWithBackends",
         F("backend/cuda/internal/gpudev/cuda.go", "backend/metal/internal/gpudev/metal.go")),
    Scan("backend/metal/internal/gpudev", "TestOutputContract_hostWritersStateTheirs",
         F("backend/cuda/internal/gpudev/cuda.go", "backend/metal/internal/gpudev/metal.go", "backend/cuda/internal/gpudev/cuda_copy.go",
           "backend/metal/internal/gpudev/metal_copy.go")),
    # cuda module
    Scan("backend/cuda", "TestKernelLocalMemoryCensus", F("backend/cuda/kernels.go", "backend/cuda/backend.go", "backend/cuda/testdata/*.ptx")),
    Scan("backend/cuda/kernels", "TestKernelFMALint_coversEmbeddedPTX", F("backend/cuda/kernels.go") + CU("backend/cuda")),
    Scan("backend/cuda/kernels", "TestKernelFMALint", CU("backend/cuda")),
    Scan("backend/cuda", "TestPTX_matchesSourcesAndBindings", F("backend/cuda/kernels.go") + NT("backend/cuda") + CU("backend/cuda")),
    Scan("backend/cuda", "TestPTXModules_coverEveryEmbed", F("backend/cuda/kernels.go")),
    Scan("backend/cuda", "TestPipelineLint_boundKernelsAreLaunched", NT("backend/cuda")),
    Scan("backend/cuda", "TestPrefillCoreAndDraftBlockCallTheShmemGuards", F("backend/cuda/prefill.go", "backend/cuda/drafter.go")),
    Scan("backend/cuda", "TestPrefillCoreCallsMropePosWindow", F("backend/cuda/prefill.go")),
    Scan("backend/cuda", "TestKV8_everyFusedPrefillLaunchPassesScales", F("backend/cuda/prefill.go", "backend/cuda/prefill_pass.go")),
    Scan("backend/cuda", "TestPrefillCoreCallsCheckPrefillShmemImg", F("backend/cuda/prefill.go")),
    Scan("backend/cuda", "TestDeclineAdvice_wiredIntoBuildResident", F("backend/cuda/backend.go")),
    Scan("backend/cuda/kernels", "TestAttnKernelDigests_tableCoversTheSource", CU("backend/cuda")),
    Scan("backend/cuda/kernels", "TestAttnDigestRig_uploadsSynchronize", F("backend/cuda/kernels/attn_digest_rig_test.go", "backend/cuda/kernels/attn_digest_test.go")),
    Scan("backend/cuda/kernels", "TestKernels_uploadsSynchronize", T("backend/cuda/kernels")),
    Scan("backend/cuda/internal/launchgeom", "TestGemvRN_matchesCuDefines", F("backend/cuda/gemv_w4a8_rn.cu")),
    Scan("backend/cuda/internal/launchgeom", "TestKQuantGemv_matchesCuDefine", F("backend/cuda/actgroup.cu")),
    Scan("backend/cuda/internal/launchgeom", "TestGemmMMA_matchesCuDefines", F("backend/cuda/gemm_w4a8_mma.cu")),
    Scan("backend/cuda/internal/launchgeom", "TestPrefillG32_matchesCuDefines", F("backend/cuda/prefill_g32.cu")),
    Scan("backend/cuda/internal/launchgeom", "TestGumbel_matchesCuDefine", F("backend/cuda/gumbel.cu")),
    Scan("backend/cuda/internal/gpudev", "TestPTXReproducible", CU("backend/cuda/internal/gpudev") + F("backend/cuda/internal/gpudev/testdata/*.ptx")),
    # Tests that read .github/workflows/ci.yml: each checks that a strided or subset run has its full form somewhere in CI.
    Scan("internal/runtime/sampling", "TestSweepCoverage_fullSweepRunsSomewhere", F(".github/workflows/ci.yml")),
    Scan("multimodal", "TestRaceSubset_runsInFullSomewhere", F(".github/workflows/ci.yml")),
    Scan("internal/runtime/kvmgr", "TestOracle_runsInFullSomewhere", F(".github/workflows/ci.yml")),
    # cmd/lint mirrors CI's staticcheck steps and the pin, writes go.work files from the root go.mod's go line, and holds the
    # cgo target's module free of platform files (fnmatch's * crosses /, so gpu/*.go is every .go file under gpu/).
    Scan("cmd/lint", "TestTargets_matchCI", F(".github/workflows/ci.yml")),
    Scan("cmd/lint", "TestPinnedVersion", F(".github/actions/staticcheck/go.mod")),
    Scan("cmd/lint", "TestFindingIsFail", F(".github/actions/staticcheck/go.mod", "go.mod")),
    Scan("cmd/lint", "TestCanaryPerPlatform", F(".github/actions/staticcheck/go.mod", "go.mod")),
    Scan("cmd/lint", "TestWorkspaceEnv", F("go.mod")),
    Scan("cmd/lint", "TestWorkspaceEnv_relativeRoot", F("go.mod")),
    Scan("cmd/lint", "TestCgoTargets_noPlatformFiles", F("backend/webgpu/*.go")),
    # SB8's codec census renders through tokenizer and multimodal and reads the tracked templates and tokenizers.
    Scan("chat", "TestCodecCensus", NT("tokenizer", "multimodal") +
         F("testdata/chat_goldens/*.json", "testdata/chat_think_goldens/*.json", "tokenizer/testdata/*.gguf")),
    Scan("chat", "TestCodecCensus_throughCodec", NT("internal/codec", "tokenizer", "multimodal") +
         F("testdata/chat_goldens/*.json", "testdata/chat_think_goldens/*.json", "tokenizer/testdata/*.gguf")),
    Scan("chat", "TestReleasing_standaloneBuildGateExists", F(".github/workflows/ci.yml", ".github/workflows/standalone-build.yml")),
    # K3 checks that no production file names an entry point its exclusions call uncalled, against the census's predicates.
    Scan("internal/diag/kernelprint", "TestCompute_reachesCensus", TREE_NT + F("docs/hardware-coverage.json")),
    Scan("internal/diag/kernelprint", "TestCompute_table", F("testdata/kernel_fingerprints.json", "go.mod", "go.sum")),
]

# Tests that read Go source but are not selected here, each with the reason. A new reader goes in SCANS or here.
EXCLUDED = {
    # The repository's own source as prompt text: not a structural check, and each is env- or heavy-gated.
    "TestKV0_prefillInvariance_real1p5B": "prompt corpus, heavy-gated",
    "TestKV0_continuationEffect_real1p5B": "prompt corpus, heavy-gated",
    "TestKV3_prefillDigest": "prompt corpus, heavy-gated",
    "TestWebGPUThetaAB": "prompt corpus, env-gated measurement",
    "TestMetalThetaAB": "prompt corpus, env-gated measurement",
    "TestThetaAB": "prompt corpus, env-gated measurement",
    "TestSpecNgram_copyOnStepVerify": "prompt corpus, env-gated",
    "TestMetalSpecPrefillRegression": "prompt corpus, env-gated",
    "TestSpecResidentPrefillRegression": "prompt corpus, env-gated",
    "TestSuffixProbe_step1": "prompt corpus, env-gated",
    "TestKV3_admissionPricesTheChunk": "prompt corpus (kv0TextTokens)",
    "TestSP2_foldCost": "prompt corpus, heavy- and env-gated measurement",
    "TestKV3_batchedPrefillPastTheSpanBound_real1p5B": "prompt corpus, heavy-gated",
    "TestKV3Tiled_forcedRouteOnGemma3": "prompt corpus (kv0TextTokens), real-checkpoint gated",
    "TestKV3Tiled_batchedEqualsPerToken_real": "prompt corpus (kv0TextTokens), real-checkpoint gated",
    "TestG46_gemma3GEMMFidelityLadder": "prompt corpus (kv0TextTokens), real-checkpoint gated",
    "TestWindowedKV_offsetsFastClass_real1p5B": "prompt corpus, heavy-gated",
    # Covered before a push by scripts/refresh_parity_hashes.sh (CLAUDE.md, "Day loop"); red on every decoder edit until then.
    "TestParityManifest_fresh": "the parity refresh's job",
    "TestParityManifest_merge": "the parity refresh's job (skips without -merge-rows)",
    # Clone or build the whole tree: minutes each, run by `go run ./cmd/gate` and CI, not the day loop.
    "TestQuick_neverDropsAGoListDependent": "whole-tree go list, cmd/gate's own suite",
    "TestQuick_realLeafMutation": "whole-tree mutation build",
    "TestQuick_realCoreMutationFailsTheGoldens": "whole-tree mutation build",
    "TestIdentity_commentOnlyChangeIsIdentical": "git clone of the tree",
    "TestIdentity_oneULPChangeFlagsExactlyTheFamiliesThatExecuteIt": "git clone of the tree",
    "TestIdentity_determinismCatchesANondeterministicDumper": "git clone of the tree",
    "TestIdentityGiw_bytesCommentOnlyChangeIsIdentical": "git clone of the tree",
    "TestIdentityGiw_headerChangeIsDifferentBytesAndIdenticalLogits": "git clone of the tree",
}


# The private notes checkout (a sibling goinfer-notes) holds the task, queue, plan and scoping docs once they leave this repository.
# A scan that reads docs/ reads them from there too, so a change in the checkout counts as a change to the path it would have had.
NOTES_AS_DOCS = (("tasks/", "docs/tasks/"), ("completed/", "docs/completed/"), ("queues/", "docs/"), ("plans/", "docs/"),
                 ("measurements/", "docs/measurements/"))


def notes_changes(root):
    """The paths changed in the notes checkout beside root (against its origin/main, or its HEAD without one; untracked files
    too), as the docs/ paths they stand for. Empty when there is no checkout, which is CI."""
    notes = os.path.join(os.path.dirname(os.path.abspath(root)), "goinfer-notes")
    if not os.path.isdir(os.path.join(notes, ".git")):
        return set()

    def git(*a):
        return subprocess.run(["git", "-C", notes, *a], capture_output=True, text=True)

    diff = git("diff", "--name-only", "origin/main")
    if diff.returncode != 0:
        diff = git("diff", "--name-only", "HEAD")
    names = set(diff.stdout.split()) | set(git("ls-files", "--others", "--exclude-standard").stdout.split())
    out = set()
    for n in names:
        for src, dst in NOTES_AS_DOCS:
            if n.startswith(src):
                out.add(dst + n[len(src):])
                break
    return out


def _selects_a_file(kind, arg, files, bydir, root):
    if kind == "workflow_paths":
        patterns = _workflow_paths(root, arg)
        return any(_path_filter(patterns, p) for p in files)
    if kind == "named_in":  # it has a file to read
        d, glob = os.path.split(arg)
        return any(fnmatch.fnmatch(b, glob) for b in bydir.get(d, ()))
    if kind in ("nt", "t", "cu"):
        return any(_matches(kind, arg, os.path.join(arg, b)) for b in bydir.get(arg, ()))
    if kind == "file":
        head = re.split(r"[*?\[]", arg, maxsplit=1)[0]
        return any(fnmatch.fnmatch(p, arg) for p in files if p.startswith(head))
    return any(_matches(kind, arg, p) for p in files)


def problems(root):
    """What in SCANS no longer matches the tree at root, a line each. A scope that selects no file: the file it names was moved,
    renamed or deleted, so a change at its new home never runs the scan. A test that is not declared in the package it is
    listed under: go test -run matches nothing there. Files are the tracked and the new ones that exist on disk."""
    ls = subprocess.run(["git", "-C", root, "ls-files", "--cached", "--others", "--exclude-standard"], capture_output=True, text=True)
    if ls.returncode != 0:  # an unlisted tree is not a tree with every file gone
        return [f"cannot list the files at {root}: {ls.stderr.strip()[:200]}"]
    files = sorted({f for f in ls.stdout.split("\n") if f and os.path.exists(os.path.join(root, f))})
    bydir = {}
    for f in files:
        d, base = os.path.split(f)
        bydir.setdefault(d, []).append(base)
    declared = {}

    def tests_of(pkg):
        if pkg not in declared:
            names = set()
            for base in bydir.get(pkg, ()):
                if base.endswith("_test.go"):
                    with open(os.path.join(root, pkg, base), encoding="utf-8", errors="replace") as fh:
                        names |= set(re.findall(r"^func ((?:Test|Fuzz)\w+)\(", fh.read(), re.M))
            declared[pkg] = names
        return declared[pkg]

    live = {}  # (kind, arg) -> whether it selects a file: many scans share a scope
    out = []
    for s in SCANS:
        what = f"{s.pkg} {'(whole package)' if s.test == WHOLE else s.test}"
        if s.test == WHOLE:
            if not tests_of(s.pkg):
                out.append(f"{what}: {s.pkg} declares no test")
        elif s.test not in tests_of(s.pkg):
            out.append(f"{what}: not declared in {s.pkg} (moved, renamed or deleted)")
        for scope in s.scopes:
            if scope not in live:
                live[scope] = _selects_a_file(*scope, files, bydir, root)
            if not live[scope]:
                out.append(f"{what}: its scope {scope[0]} {scope[1]!r} selects no file (moved, renamed or deleted)")
    return out


def select(changed, root=None):
    """{package dir: sorted test names} for the scan tests that read any of the changed paths, in the tree at root (the
    repository this file is in, by default). WHOLE among a package's names stands for its whole test run."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    memo, out = {}, {}
    for s in SCANS:
        if any(_reads(k, a, p, root, memo) for k, a in s.scopes for p in changed):
            out.setdefault(s.pkg, set()).add(s.test)
    return {pkg: sorted(t) for pkg, t in out.items()}
