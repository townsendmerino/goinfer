"""The source-scan tests touched.py must run when a file they read changes.

A source-scan test reads the repository's own files (Go source, kernel source) and checks a property across them: a census
of dispatch sites, a switch's case labels, every route's guard. Selecting tests by the test file beside a changed file misses
it, because the test lives elsewhere, often in another package or module. SCANS lists each one with the paths it reads;
select() maps a set of changed paths to the tests to run. scripts/test_touched.py fails on a test that reads Go source and is
in neither SCANS nor EXCLUDED, so a new scanner cannot be missed silently.
"""
import fnmatch
import os

SKIP_DIRS = ("testdata/", "node_modules/", ".git/", "site/")


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
    # MM1's decision census pins every fit decision: a change to the fit code or the machine zoo runs it.
    Scan("decoder", "TestDecisionCensus", F("decoder/fitguard.go", "decoder/fitplan.go", "decoder/fitquery.go", "decoder/prefill_budget.go",
                                            "decoder/weightbytes.go", "decoder/residentneed.go", "decoder/kvresolve.go", "decoder/kvpolicy.go",
                                            "decoder/metal_fitctx.go", "decoder/memprobe.go", "testdata/machines/*.json", "testdata/mm1_census.json.gz") +
         NT("internal/mem")),
    # VK2's census: each family's own list covers its registry adapter's file and the arch_*.go files it reaches.
    Scan("decoder", "TestParityManifest_adapterFilesOwned", NT("decoder") + F("testdata/parity/*.json")),
    # VK2's forward census: every file the forward reaches is in a hash set or an own list.
    Scan("decoder", "TestParityManifest_coreClosed", NT("decoder") + F("testdata/parity/*.json")),
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
    Scan("decoder", "TestGoldenNames_matchTheFileOnDisk", TREE_T),
    Scan("decoder", "TestOptionGrid_cellsCarryEvidence", TREE_GO),
    Scan("decoder", "TestAssetRegistry_usedByExists", TREE_T + F("testdata/assets.json")),
    Scan("decoder", "TestAssetRegistry_noDirectReads", TREE_T + F("testdata/assets.json")),
    # Two computations of one key over the same files: only the two implementations or the manifest can split them.
    Scan("decoder", "TestParityKey_matchesFreshDepsHash", F("decoder/parity_manifest_test.go", "testdata/parity/*.json") + NT("internal/paritykey")),
    # The fixture censuses walk testdata/: a new or changed model fixture must be listed or excluded in each (8ec704c3).
    Scan("decoder", "TestSerializeCensus_everyFixtureIsListedOrExcluded", F("testdata/*/config.json")),
    Scan("decoder", "TestSeams_subCaptureRefusedWhereGateSCannotRun", F("testdata/*/config.json")),
    Scan("decoder", "TestParityEmit_methodVocabulary", TREE_T),
    Scan("decoder", "TestResidentMoECap_pinsKernelConstant", F("gpu/moe.go", "cuda/moe.cu")),
    Scan("decoder", "TestResidentMoECap_runtimeDeclineReadsTheDeclaration", F("gpu/residency.go")),
    # internal/serveapp
    Scan("internal/serveapp", "TestBanner_contextIsTheEnforcedWindow", F("internal/serveapp/banner.go", "internal/serveapp/main.go")),
    Scan("internal/serveapp", "TestServeVisionChat_appliesExtractionRuleBeforeThePromptIsBuilt", F("internal/serveapp/vision_serve.go")),
    Scan("internal/serveapp", "TestServe_everyTokenizingRouteGuardsItsInputSize", NT("internal/serveapp")),
    Scan("internal/serveapp", "TestDecoderEmbedder_boundComesFromTheModelNotZero", F("internal/serveapp/decoder_embedder.go")),
    Scan("internal/serveapp", "TestAnthropic_decodeUsesTheSharedShaping", F("internal/serveapp/anthropic.go")),
    Scan("internal/serveapp", "TestSessions_persistedStateIsOwnerOnly", F("internal/serveapp/sessions.go")),
    Scan("internal/serveapp", "TestAdminLoad_racedDuplicateIsClosed", F("internal/serveapp/admin.go", "internal/serveapp/webui.go")),
    Scan("internal/serveapp", "TestStartDrain_savesSessionsUnderTheLRULock", F("internal/serveapp/liveness.go")),
    Scan("internal/serveapp", "TestAdminLoad_requestQuantIsTheExplicitOne", F("internal/serveapp/admin.go")),
    Scan("internal/serveapp", "TestLookupLockedOnlyViaResolve", NT("internal/serveapp")),
    Scan("internal/serveapp", "TestStreamSurfaces_allSendUsageBeforeDone",
         F("internal/serveapp/openai.go", "internal/serveapp/tools.go", "internal/serveapp/vision_serve.go")),
    Scan("internal/serveapp", "TestStreamTokens_decodesAsAContinuation", F("internal/serveapp/openai.go")),
    Scan("internal/serveapp", "TestServe_prepareEnforcesThePublishedWindow", F("internal/serveapp/openai.go")),
    Scan("internal/serveapp", "TestDriveUsesRequestContext", NT("internal/serveapp")),
    Scan("internal/serveapp", "TestRequestLog_wiredIntoTheGenerationRoutes", F("internal/serveapp/main.go")),
    Scan("internal/serveapp", "TestRespondTools_callsMaybeStoreWithTheToolCalls", F("internal/serveapp/responses.go")),
    Scan("internal/serveapp", "TestSSE_everyStreamingDriveSiteHeartbeats",
         F("internal/serveapp/openai.go", "internal/serveapp/anthropic_stream.go", "internal/serveapp/vision_serve.go",
           "internal/serveapp/responses.go")),
    Scan("internal/serveapp", "TestSSE_everyBufferedStreamSiteHeartbeats", NT("internal/serveapp")),
    Scan("internal/serveapp", "TestSupportTable_everyFamilyHasALoader", F("internal/serveapp/main.go", "docs/multimodal.md")),
    Scan("internal/serveapp", "TestServe_noRouteEncodesARenderedPrompt", NT("internal/serveapp")),
    Scan("internal/serveapp", "TestVision_imageBlockNewlinesMatchTheRealProcessors",
         F("internal/serveapp/vision_multi.go", "internal/serveapp/vision_serve.go")),
    Scan("internal/serveapp", "TestWebUI_rootRouteIsUnauthenticated", F("internal/serveapp/main.go")),
    Scan("internal/serveapp", "TestWebUI_pageRoutesExistOnlyUnderWebFlag", F("internal/serveapp/main.go")),
    Scan("internal/serveapp", "TestWebUI_listAndPullAreWrappedInSameOrigin", F("internal/serveapp/main.go")),
    # other root-module packages
    Scan("internal/cliutil", "TestLdflagsTargetsExist", NT("internal/chatapp", "internal/serveapp")),
    Scan("internal/fitcmd", "TestDefaultCtx_agreesWithTheCudaPlannerAndHoldsAnAgentTurn", F("cuda/resident.go", "internal/fitcmd/fit.go")),
    Scan("internal/fitcmd", "TestRun_closesFirstLoadBeforeMeasuring", F("internal/fitcmd/fit.go")),
    Scan("internal/hwcensus", "TestHardwareCensus", TREE_NT + F("docs/hardware-coverage.json")),
    Scan("internal/mem/probe", "TestOnlyProbeReadsOSMemory", TREE_NT),
    Scan("internal/mem", "TestImports", NT("internal/mem")),
    Scan("internal/mem", "TestSolve_pure", NT("internal/mem")),
    Scan("internal/mem/probe", "TestProbeImports", NT("internal/mem/probe")),
    Scan("internal/layout", "TestBackendTests_decoderImportsOnlyFall", TREE_GO),
    # SB13: every goinfer-to-goinfer import against internal/layout/imports.json, and the root go.mod against its rules.
    Scan("internal/layout", "TestLayout_arrows", TREE_GO + F("internal/layout/imports.json", "go.mod")),
    Scan("internal/prezerolint", "TestPreZero_noDeadZeroingBeforeOverwriteKernels", TREE_NT),
    Scan("internal/movedpaths", "TestNoMovedAikitPath", TREE_GO + F("go.mod", "*/go.mod", "go.sum", "*/go.sum")),
    Scan("pull", "TestResolve_callsResolveOfflineBeforeCheckAccess", F("pull/resolve.go")),
    Scan("examples/structured", "TestUseFromGoDoc_codeIsFromExamples", F("examples/*/main.go", "docs/use-from-go.md")),
    Scan("cmd/gate", "TestGPU_metalPrefillCellChecksVacuous", F("cmd/gate/gpu.go")),
    Scan("cmd/gate", "TestServeRoutes_findsTheRealOnes", NT("internal/serveapp")),
    Scan("cmd/gate", "TestHeavyPartition_cudaTreeFitsTheTable", T("cuda")),
    Scan("cmd/gate", "TestGPU_isolatedGroupIsDerivedFromTheMarker", T("cuda")),
    Scan("cmd/gate", "TestQuick_slowPatternsNameRealTests", T("gpu", "cuda", "metal")),
    Scan("cmd/gate", "TestWebGPUGateIsListedOrExplicitlyNotRequired", T("gpu")),
    Scan("cmd/gate", "TestRealckptGateIsListedOrExplicitlyNotRequired", T("decoder")),
    Scan("cmd/gate", "TestRealckptRunReachesEveryScannedAndRequiredGate", T("decoder")),
    Scan("cmd/gate", "TestRealckptCellCanReachEveryGate", T("decoder")),
    Scan("cmd/gate", "TestGateRunFilter_narrowsTheRealckptPattern", T("decoder")),
    Scan("cmd/gate", "TestRealckptRunIsAdditiveOverTheLegacyPattern", T("decoder")),
    Scan("cmd/gate", "TestMetalGateIsListedOrExplicitlyNotRequired", T("metal")),
    # demo/agent module
    Scan("demo/agent/agent", "TestAgentWeb_hardening", F("demo/agent/cmd/agent-web/main.go")),
    Scan("demo/agent/agent", "TestAgentSession_encodesSegmentsNotRenderedText", F("demo/agent/agent/agent.go")),
    Scan("demo/agent/agent", "TestTurnImage_wiresSpliceImageBlockBeforeEncoding", F("demo/agent/agent/agent.go")),
    # site module
    Scan("site/internal/site", "TestHome_heroCodeIsFromTheExample", F("examples/structured/main.go")),
    # gpu module
    Scan("gpu", "TestNemotronBlockKind_hasADefault", F("gpu/residency.go")),
    Scan("gpu", "TestEnsureAttnWide_guardsOnTheLastPipeline", F("gpu/attention.go")),
    Scan("gpu", "TestGPU_deviceBoundaryErrorsAreNotDropped", F("gpu/decoderunner.go", "gpu/residency.go")),
    Scan("gpu", "TestResidency_kvPrecisionDeclineCoversTheNonGenericBranches", F("gpu/residency.go")),
    # metal module
    Scan("metal", "TestGptOssBiasDispatches_alwaysIndexedByRealExpert", F("metal/moe.go")),
    Scan("metal/internal/gpudev", "TestOutputContract_newBufferDocsAgreeWithBackends",
         F("cuda/internal/gpudev/cuda.go", "metal/internal/gpudev/metal.go")),
    Scan("metal/internal/gpudev", "TestOutputContract_hostWritersStateTheirs",
         F("cuda/internal/gpudev/cuda.go", "metal/internal/gpudev/metal.go", "cuda/internal/gpudev/cuda_copy.go",
           "metal/internal/gpudev/metal_copy.go")),
    # cuda module
    Scan("cuda", "TestKernelLocalMemoryCensus", F("cuda/kernels.go", "cuda/backend.go", "cuda/testdata/*.ptx")),
    Scan("cuda", "TestKernelFMALint_coversEmbeddedPTX", F("cuda/kernels.go") + CU("cuda")),
    Scan("cuda", "TestKernelFMALint", CU("cuda")),
    Scan("cuda", "TestPTX_matchesSourcesAndBindings", F("cuda/kernels.go") + NT("cuda") + CU("cuda")),
    Scan("cuda", "TestPTXModules_coverEveryEmbed", F("cuda/kernels.go")),
    Scan("cuda", "TestPipelineLint_boundKernelsAreLaunched", NT("cuda")),
    Scan("cuda", "TestPrefillCoreAndDraftBlockCallTheShmemGuards", F("cuda/prefill.go", "cuda/drafter.go")),
    Scan("cuda", "TestPrefillCoreCallsMropePosWindow", F("cuda/prefill.go")),
    Scan("cuda", "TestPrefillCoreCallsCheckPrefillShmemImg", F("cuda/prefill.go")),
    Scan("cuda", "TestDeclineAdvice_wiredIntoBuildResident", F("cuda/backend.go")),
    Scan("cuda/kernels", "TestAttnKernelDigests_tableCoversTheSource", CU("cuda")),
    Scan("cuda/internal/gpudev", "TestPTXReproducible", CU("cuda/internal/gpudev") + F("cuda/internal/gpudev/testdata/*.ptx")),
    # K3 checks that no production file names an entry point its exclusions call uncalled, against the census's predicates.
    Scan("internal/kernelprint", "TestCompute_reachesCensus", TREE_NT + F("docs/hardware-coverage.json")),
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
