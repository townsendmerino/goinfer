"""The source-scan tests touched.py must run when a file they read changes.

A source-scan test reads the repository's own files (Go source, kernel source) and checks a property across them: a census
of dispatch sites, a switch's case labels, every route's guard. Selecting tests by the test file beside a changed file misses
it, because the test lives elsewhere, often in another package or module. SCANS lists each one with the paths it reads;
select() maps a set of changed paths to the tests to run. scripts/test_touched.py fails on a test that reads Go source and is
in neither SCANS nor EXCLUDED, so a new scanner cannot be missed silently.
"""
import fnmatch
import os

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
    if kind == "tree_nt":
        return not skipped and path.endswith(".go") and not path.endswith("_test.go")
    if kind == "tree_t":
        return not skipped and path.endswith("_test.go")
    raise ValueError(kind)


SCANS = [
    # decoder
    Scan("decoder", "TestDispatchCensus", NT("decoder")),
    # MM1's decision census pins every fit decision: a change to the fit code, the machine zoo or a fixture it reads runs it. Every
    # censusFixtures entry has "tiny" in its name (TestCensusFixtures_areTheCommittedOnes).
    Scan("decoder", "TestDecisionCensus", F("decoder/fitguard.go", "decoder/fitplan.go", "decoder/fitquery.go", "decoder/prefill_budget.go",
                                            "decoder/weightbytes.go", "decoder/residentneed.go", "decoder/kvresolve.go", "decoder/kvpolicy.go",
                                            "decoder/metal_fitctx.go", "decoder/memprobe.go", "testdata/machines/*.json", "testdata/mm1_census.json.gz",
                                            "decoder/mm1_census_fixtures_test.go", "testdata/*tiny*", "decoder/testdata/*tiny*",
                                            "decoder/selftestdata/*tiny*", "internal/vision/testdata/*tiny*") +
         NT("internal/runtime/mem")),
    # The census reads a declared fixture list, which must match the committed fixtures: a fixture added or removed runs it.
    Scan("decoder", "TestCensusFixtures_areTheCommittedOnes", F("decoder/mm1_census_fixtures_test.go", "testdata/*", "decoder/testdata/*",
                                                                "decoder/selftestdata/*", "internal/vision/testdata/*")),
    # VK2's census: each family's own list covers its registry adapter's file and the arch_*.go files it reaches.
    Scan("decoder", "TestParityManifest_adapterFilesOwned", NT("decoder") + F("testdata/parity/*.json")),
    # VK2's forward census: every file the forward reaches is in a hash set or an own list.
    Scan("decoder", "TestParityManifest_coreClosed", NT("decoder") + F("testdata/parity/*.json")),
    # A record that hashes a test file running through textParity hashes the helper too.
    Scan("decoder", "TestTextParity_hashedWithItsUsers", T("decoder") + F("testdata/parity/*.json")),
    Scan("decoder", "TestTranscode_writesViaTempThenRenames", F("decoder/sidecar.go")),
    Scan("decoder", "TestTranscodeDir_writesViaTempThenRenames", F("decoder/sidecar.go")),
    Scan("decoder", "TestGGUFConfig_everyArchitectureReadsMaxPositions", F("decoder/gguf.go")),
    Scan("decoder", "TestStreamableFamilyClosures_onlyReadPerLayerTensors", F("decoder/gguf.go")),
    Scan("decoder", "TestEmbedInt4_noLoaderPinsTheHeadItself", NT("decoder")),
    Scan("decoder", "TestProgram_noFamilyLoopAndEveryFamilyWalks", NT("decoder")),
    Scan("decoder", "TestFP8_shapeCheckDistinguishesTranspose", F("decoder/fp8.go")),
    Scan("decoder", "TestGoDoc_listsEveryFieldOfOptionsAndSamplingParams", NT("decoder")),
    Scan("decoder", "TestBackendBanner_usesTheReport", TREE_NT),
    Scan("decoder", "TestEnvVars_docAndCodeAgree", TREE_GO + F("docs/env-vars.md", "testdata/env_reads.txt")),
    Scan("internal/config", "TestInventory", F("decoder/knobs.go", "decoder/model.go", "testdata/env_reads.txt", "docs/flags.meta.json") + NT("internal/app/loadflags")),
    Scan("internal/config", "TestInventory_plants", F("decoder/knobs.go", "decoder/model.go", "testdata/env_reads.txt", "docs/flags.meta.json") + NT("internal/app/loadflags")),
    Scan("internal/config", "TestEnvVarsDoc", F("docs/env-vars.md", "docs/env-vars.meta.json")),
    Scan("internal/config", "TestEnvVarsDoc_plants", F("docs/env-vars.meta.json")),
    Scan("internal/app/serveapp", "TestOneMerge", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestFlagsDoc", F("docs/flags.md", "docs/flags.meta.json") + NT("internal/app/serveapp")),
    Scan("decoder", "TestGoldenNames_matchTheFileOnDisk", TREE_T),
    Scan("decoder", "TestOptionGrid_cellsCarryEvidence", TREE_GO),
    Scan("decoder", "TestAssetRegistry_usedByExists", TREE_T + F("testdata/assets.json", "decoder/assets.go")),
    Scan("decoder", "TestAssetRegistry_noDirectReads", TREE_T + F("testdata/assets.json", "decoder/assets.go")),
    Scan("decoder", "TestAssetRegistry_resolvesFromEveryCallerDir", TREE_T + F("testdata/assets.json", "decoder/assets.go")),
    # Two computations of one key over the same files: only the two implementations or the manifest can split them.
    Scan("decoder", "TestParityKey_matchesFreshDepsHash", F("decoder/parity_manifest_test.go", "testdata/parity/*.json", "testdata/kernel_fingerprints.json") + NT("internal/guard/paritykey")),
    Scan("decoder", "TestParityKey_shadowMatchesDecoder", F("testdata/loader_identity.json", "decoder/parity_manifest_test.go", "testdata/parity/*.json", "testdata/kernel_fingerprints.json", "testdata/program_fingerprints.json") + NT("internal/guard/paritykey", "decoder")),
    Scan("decoder", "TestParityPrograms_mapping", F("testdata/parity/*.json", "testdata/program_fingerprints.json", "testdata/*/config.json") + NT("decoder")),
    Scan("decoder", "TestParityLoaders_mapping", F("testdata/parity/*.json", "testdata/loader_identity.json") + NT("internal/guard/paritykey")),
    Scan("internal/guard/paritykey", "TestBuilderFile_programTypesDeclareOnly", F("internal/runtime/program/program.go")),
    # The fixture censuses walk testdata/: a new or changed model fixture must be listed or excluded in each (8ec704c3).
    Scan("decoder", "TestSerializeCensus_everyFixtureIsListedOrExcluded", F("testdata/*/config.json")),
    Scan("decoder", "TestSeams_subCaptureRefusedWhereGateSCannotRun", F("testdata/*/config.json")),
    Scan("decoder", "TestParityEmit_methodVocabulary", TREE_T),
    Scan("decoder", "TestResidentMoECap_pinsKernelConstant", F("backend/webgpu/moe.go", "backend/cuda/moe.cu")),
    Scan("decoder", "TestResidentMoECap_runtimeDeclineReadsTheDeclaration", F("backend/webgpu/residency.go")),
    # internal/app/serveapp
    Scan("internal/app/serveapp", "TestBanner_contextIsTheEnforcedWindow", F("internal/app/serveapp/banner.go", "internal/app/serveapp/main.go")),
    Scan("internal/app/serveapp", "TestServeVisionChat_appliesExtractionRuleBeforeThePromptIsBuilt", F("internal/app/serveapp/vision_serve.go")),
    Scan("internal/app/serveapp", "TestServe_everyTokenizingRouteGuardsItsInputSize", NT("internal/app/serveapp")),
    # SB7: a loadedModel's model handle is set only through handleOf.
    Scan("engine", "TestSurface_exportsOnlyItsContract", NT("engine", "engine/enginetest")),
    Scan("internal/app/serveapp", "TestModelHandle_assignedThroughHandleOf", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestSessions_persistedStateIsOwnerOnly", F("internal/app/serveapp/sessions.go")),
    Scan("internal/app/serveapp", "TestAdminLoad_racedDuplicateIsClosed", F("internal/app/serveapp/admin.go", "internal/app/serveapp/webui.go")),
    Scan("internal/app/serveapp", "TestStartDrain_savesSessionsUnderTheLRULock", F("internal/app/serveapp/liveness.go")),
    Scan("internal/app/serveapp", "TestAdminLoad_requestQuantIsTheExplicitOne", F("internal/app/serveapp/admin.go")),
    Scan("internal/app/serveapp", "TestLookupLockedOnlyViaResolve", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestStreamSurfaces_allSendUsageBeforeDone",
         F("internal/app/serveapp/openai.go", "internal/app/serveapp/tools.go", "internal/app/serveapp/vision_serve.go")),
    Scan("internal/app/serveapp", "TestStreamTokens_decodesAsAContinuation", F("internal/app/serveapp/openai.go")),
    Scan("internal/app/serveapp", "TestServe_prepareEnforcesThePublishedWindow", F("internal/app/serveapp/openai.go")),
    Scan("internal/app/serveapp", "TestDriveUsesRequestContext", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestRespondTools_callsMaybeStoreWithTheToolCalls", F("internal/app/serveapp/responses.go")),
    Scan("internal/app/serveapp", "TestSSE_everyStreamingDriveSiteHeartbeats",
         F("internal/app/serveapp/openai.go", "internal/app/serveapp/anthropic_stream.go", "internal/app/serveapp/vision_serve.go",
           "internal/app/serveapp/responses.go")),
    Scan("internal/app/serveapp", "TestSSE_everyBufferedStreamSiteHeartbeats", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestSupportTable_everyFamilyHasALoader", F("internal/app/serveapp/main.go", "docs/multimodal.md")),
    Scan("internal/app/serveapp", "TestServe_noRouteEncodesARenderedPrompt", NT("internal/app/serveapp")),
    Scan("internal/app/serveapp", "TestVision_imageBlockNewlinesMatchTheRealProcessors",
         F("internal/app/serveapp/vision_multi.go", "internal/app/serveapp/vision_serve.go")),
    Scan("internal/app/serveapp", "TestWebUI_rootRouteIsUnauthenticated", F("internal/app/serveapp/main.go")),
    Scan("internal/app/serveapp", "TestWebUI_pageRoutesExistOnlyUnderWebFlag", F("internal/app/serveapp/main.go")),
    Scan("internal/app/serveapp", "TestWebUI_listAndPullAreWrappedInSameOrigin", F("internal/app/serveapp/main.go")),
    # other root-module packages
    Scan("internal/app/cliutil", "TestLdflagsTargetsExist", NT("internal/app/chatapp", "internal/app/serveapp")),
    Scan("internal/app/fitcmd", "TestDefaultCtx_agreesWithTheCudaPlannerAndHoldsAnAgentTurn", F("backend/cuda/resident.go", "internal/app/fitcmd/fit.go")),
    Scan("internal/app/fitcmd", "TestRun_closesFirstLoadBeforeMeasuring", F("internal/app/fitcmd/fit.go")),
    Scan("internal/diag/hwcensus", "TestHardwareCensus", TREE_NT + F("docs/hardware-coverage.json")),
    Scan("internal/runtime/mem/probe", "TestOnlyProbeReadsOSMemory", TREE_NT),
    Scan("internal/runtime/mem", "TestImports", NT("internal/runtime/mem")),
    Scan("backend/cuda", "TestPlanSlack_oneCopy", NT("backend/cuda")),
    Scan("backend/cuda", "TestLedger_everyAllocationNamesALine", NT("backend/cuda")),
    Scan("internal/runtime/mem", "TestSolve_pure", NT("internal/runtime/mem")),
    Scan("internal/runtime/mem", "TestNoGuardThresholdOutsideMem", TREE_NT),
    Scan("internal/runtime/mem/probe", "TestProbeImports", NT("internal/runtime/mem/probe")),
    Scan("internal/layout", "TestBackendTests_decoderImportsOnlyFall", TREE_GO),
    # SB7: internal/app/serveapp test files importing decoder may only fall.
    Scan("internal/layout", "TestServeappTests_decoderImportsOnlyFall", TREE_GO),
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
    Scan("cmd/gate", "TestServeRoutes_findsTheRealOnes", NT("internal/app/serveapp")),
    Scan("decoder", "TestPolicyTests_loadedRatchet", F("decoder/spec_*_test.go", "decoder/speculative*_test.go", "decoder/session*_test.go", "testdata/sb5/loaded_policy_tests.txt")),
    Scan("decoder", "TestPolicyStreams_census", F("testdata/sb5/policy_streams.json")),
    Scan("cmd/gate", "TestHeavyPartition_cudaTreeFitsTheTable", T("backend/cuda")),
    Scan("cmd/gate", "TestGPU_isolatedGroupIsDerivedFromTheMarker", T("backend/cuda")),
    Scan("cmd/gate", "TestQuick_slowPatternsNameRealTests", T("backend/webgpu", "backend/cuda", "backend/metal")),
    Scan("cmd/gate", "TestWebGPUGateIsListedOrExplicitlyNotRequired", T("backend/webgpu")),
    Scan("cmd/gate", "TestRealckptGateIsListedOrExplicitlyNotRequired", T("decoder")),
    Scan("cmd/gate", "TestRealckptRunReachesEveryScannedAndRequiredGate", T("decoder")),
    Scan("cmd/gate", "TestRealckptCellCanReachEveryGate", T("decoder")),
    Scan("cmd/gate", "TestGateRunFilter_narrowsTheRealckptPattern", T("decoder")),
    Scan("cmd/gate", "TestGraderCensus_everySubcommand", F("cmd/gate/main.go", "docs/measurements/sb12-2026-10.md")),
    Scan("cmd/gate", "TestGraders_arePure", NT("cmd/gate")),
    Scan("cmd/gate", "TestGraderGoldens", F("testdata/gate/goldens/*", "docs/measurements/giw-v16-transcode-2026-10/*")),
    Scan("cmd/gate", "TestRealckptRunIsAdditiveOverTheLegacyPattern", T("decoder")),
    Scan("cmd/gate", "TestMetalGateIsListedOrExplicitlyNotRequired", T("backend/metal")),
    # demo/agent module
    Scan("demo/agent/agent", "TestAgentWeb_hardening", F("demo/agent/cmd/agent-web/main.go")),
    Scan("demo/agent/agent", "TestAgentSession_encodesSegmentsNotRenderedText", F("demo/agent/agent/agent.go")),
    Scan("demo/agent/agent", "TestTurnImage_wiresSpliceImageBlockBeforeEncoding", F("demo/agent/agent/agent.go")),
    # site module
    Scan("apps/site/internal/site", "TestHome_heroCodeIsFromTheExample", F("apps/examples/structured/main.go")),
    # gpu module
    Scan("backend/webgpu", "TestNemotronBlockKind_hasADefault", F("backend/webgpu/residency.go")),
    Scan("backend/webgpu/internal/kernels", "TestLedger_everyBufferIsLeased", NT("backend/webgpu", "backend/webgpu/internal/kernels")),
    Scan("backend/webgpu", "TestEnsureAttnWide_guardsOnTheLastPipeline", F("backend/webgpu/attention.go")),
    Scan("backend/webgpu", "TestGPU_deviceBoundaryErrorsAreNotDropped", F("backend/webgpu/decoderunner.go", "backend/webgpu/residency.go")),
    Scan("backend/webgpu", "TestResidency_kvPrecisionDeclineCoversTheNonGenericBranches", F("backend/webgpu/residency.go")),
    # metal module
    Scan("backend/metal", "TestGptOssBiasDispatches_alwaysIndexedByRealExpert", F("backend/metal/moe.go")),
    Scan("backend/metal", "TestLedger_everyMetalAllocationNamesALine", NT("backend/metal", "backend/metal/internal/gpudev", "backend/metal/internal/qwenmetal", "backend/metal/internal/visionmetal")),
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
    Scan("backend/cuda", "TestPrefillCoreCallsCheckPrefillShmemImg", F("backend/cuda/prefill.go")),
    Scan("backend/cuda", "TestDeclineAdvice_wiredIntoBuildResident", F("backend/cuda/backend.go")),
    Scan("backend/cuda/kernels", "TestAttnKernelDigests_tableCoversTheSource", CU("backend/cuda")),
    Scan("backend/cuda/kernels", "TestAttnDigestRig_uploadsSynchronize", F("backend/cuda/kernels/attn_digest_rig_test.go", "backend/cuda/kernels/attn_digest_test.go")),
    Scan("backend/cuda/kernels", "TestKernels_uploadsSynchronize", T("backend/cuda/kernels")),
    Scan("backend/cuda/internal/launchgeom", "TestGemvRN_matchesCuDefines", F("backend/cuda/gemv_w4a8_rn.cu")),
    Scan("backend/cuda/internal/launchgeom", "TestKQuantGemv_matchesCuDefine", F("backend/cuda/actgroup.cu")),
    Scan("backend/cuda/internal/launchgeom", "TestGemmMMA_matchesCuDefines", F("backend/cuda/gemm_w4a8_mma.cu")),
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


def select(changed):
    """{package dir: sorted test names} for the scan tests that read any of the changed paths."""
    out = {}
    for s in SCANS:
        if any(_matches(k, a, p) for k, a in s.scopes for p in changed):
            out.setdefault(s.pkg, set()).add(s.test)
    return {pkg: sorted(t) for pkg, t in out.items()}
