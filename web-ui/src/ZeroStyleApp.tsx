import { useState, useEffect, useCallback } from "react";
import { MonacoEditor } from "./components/MonacoEditor";
import ProblemPanel from "./components/ProblemPanel";
import ResizablePanels from "./components/ResizablePanels";
import VerticalResizablePanels from "./components/VerticalResizablePanels";
import { TestResultsPanel } from "./components/TestResultsPanel";
import { getProblemById, Problem } from "./utils/problemLoader";
import "./App.css";
import { Footer } from "./components/Footer";
import { Footer as EditorFooter } from "./editor/Footer";
import { useSubmitSolution } from "./hooks/api/useSubmitSolution";
import { SubmissionModal } from "./components/SubmissionModal";
import { useSimulation } from "./simulation/hooks/useSimulation";
import { AttributionLog } from "./attribution/log";
import * as monaco from "monaco-editor";


// UI Configuration - embedded directly where used
const UI_CONFIG = {
  PANELS: {
    DEFAULT_LEFT_WIDTH: 600,
    MIN_LEFT_WIDTH: 300,
    MAX_LEFT_WIDTH: 800,
  },
} as const;

// Get parameters from URL
function getZeroStyleParams(): {
  episode?: string;
  timestep?: number;
  problemId?: string;
  startingCode?: string;
  attributionLog?: AttributionLog;
  maxAssistantActions?: number;
  maxHumanActions?: number;
  assistantNoiseProb?: number;
  assistantNoiseTopK?: number;
} {
  const urlParams = new URLSearchParams(window.location.search);
  const episodeParam = urlParams.get('episode');
  const timestepParam = urlParams.get('timestep');
  const problemIdParam = urlParams.get('problemId');
  const startingCodeParam = urlParams.get('startingCode');
  const attributionLogParam = urlParams.get('attributionLog');
  const maxAssistantActionsParam = urlParams.get('maxAssistantActions');
  const maxHumanActionsParam = urlParams.get('maxHumanActions');
  const assistantNoiseProbParam = urlParams.get('assistantNoiseProb');
  const assistantNoiseTopKParam = urlParams.get('assistantNoiseTopK');
  
  const episode = episodeParam || undefined;
  const timestep = timestepParam ? parseInt(timestepParam, 10) : undefined;
  const problemId = problemIdParam || undefined;
  const startingCode = startingCodeParam ? decodeURIComponent(startingCodeParam) : undefined;
  const maxAssistantActions = maxAssistantActionsParam
    ? Math.max(1, parseInt(maxAssistantActionsParam, 10))
    : undefined;
  const maxHumanActions = maxHumanActionsParam
    ? Math.max(0, parseInt(maxHumanActionsParam, 10))
    : undefined;
  const assistantNoiseProb = assistantNoiseProbParam
    ? Math.min(Math.max(parseFloat(assistantNoiseProbParam), 0), 1)
    : undefined;
  const assistantNoiseTopK = assistantNoiseTopKParam
    ? Math.max(1, parseInt(assistantNoiseTopKParam, 10))
    : undefined;
  
  let attributionLog: AttributionLog | undefined;
  if (attributionLogParam) {
    try {
      attributionLog = JSON.parse(decodeURIComponent(attributionLogParam));
    } catch (error) {
      console.error("Failed to parse attributionLog from URL:", error);
    }
  }
  
  console.log(`📋 ZeroStyle Configuration:`);
  console.log(`   Episode: ${episode || 'Not specified'}`);
  console.log(`   Timestep: ${timestep || 'Not specified'}`);
  console.log(`   Problem ID: ${problemId || 'Not specified'}`);
  console.log(`   Starting Code: ${startingCode ? 'Provided' : 'Not provided'}`);
  console.log(`   Attribution Log: ${attributionLog ? 'Provided' : 'Not provided'}`);
  console.log(`   Max Assistant Actions: ${maxAssistantActions || 'Default'}`);
  console.log(`   Max Human Actions: ${maxHumanActions ?? 'Default'}`);
  console.log(`   Assistant Noise Probability: ${assistantNoiseProb ?? 'Default'}`);
  console.log(`   Assistant Noise TopK: ${assistantNoiseTopK ?? 'Default'}`);
  
  return {
    episode,
    timestep,
    problemId,
    startingCode,
    attributionLog,
    maxAssistantActions,
    maxHumanActions,
    assistantNoiseProb,
    assistantNoiseTopK,
  };
}

function ZeroStyleApp() {
  const [problem, setProblem] = useState<Problem | null>(null);
  const [initCode, setInitCode] = useState("");
  const [code, setCode] = useState("");
  const [editor, setEditor] = useState<monaco.editor.IStandaloneCodeEditor | null>(null);
  const [isProblemLoading, setIsProblemLoading] = useState(false);
  const [attributionLog, setAttributionLog] = useState<AttributionLog | null>(null);
  const [initialTimestep, setInitialTimestep] = useState<number | undefined>(undefined);
  const [sourceEpisode, setSourceEpisode] = useState<string | undefined>(undefined);
  const [maxAssistantActions, setMaxAssistantActions] = useState<number>(2);
  const [maxHumanActions, setMaxHumanActions] = useState<number>(1);
  const [assistantNoiseProb, setAssistantNoiseProb] = useState<number>(0.05);
  const [assistantNoiseTopK, setAssistantNoiseTopK] = useState<number>(3);
  const [error, setError] = useState<string | null>(null);
  
  const { submitSolution, submissionState } = useSubmitSolution({
    code,
    problem,
  });
  const [isSubmissionModalOpen, setIsSubmissionModalOpen] = useState(false);
  const [manualAssistantPaused, setManualAssistantPaused] = useState(false);

  const assistantPaused =
    submissionState.type === "loading" ||
    isSubmissionModalOpen ||
    manualAssistantPaused;

  // Show test results panel after submission modal is closed
  const showTestResults = submissionState.type === "data" && !isSubmissionModalOpen;

  // Simulation functionality
  const { simulationState, startSimulation } = useSimulation({
    editor,
    problemId: problem?.id,
    attributionLog: attributionLog || undefined,
    initialTimestep: initialTimestep,
  });
  const [showSimulationNotice, setShowSimulationNotice] = useState(false);

  useEffect(() => {
    setShowSimulationNotice(simulationState.isRunning);
  }, [simulationState.isRunning]);

  useEffect(() => {
    if (submissionState.type === "data") {
      setIsSubmissionModalOpen(true);
    }
  }, [submissionState]);

  const loadProblem = useCallback(async (problemId: string) => {
    try {
      setIsProblemLoading(true);
      setError(null);
      const problem = await getProblemById(problemId);
      if (!problem) {
        throw new Error(`Problem with ID ${problemId} not found`);
      }
      setProblem(problem);
      return problem;
    } catch (error) {
      console.error("Failed to load problem:", error);
      setError(error instanceof Error ? error.message : "Failed to load problem");
      return null;
    } finally {
      setIsProblemLoading(false);
    }
  }, []);

  const loadEpisodeData = useCallback(async (episode: string, timestep: number) => {
    try {
      setIsProblemLoading(true);
      setError(null);

      console.log(`Loading episode ${episode} at timestep ${timestep}`);

      // Fetch episode data from the API route
      const response = await fetch(`/api/episode-data?episode=${encodeURIComponent(episode)}&timestep=${timestep}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP ${response.status}: Failed to load episode data`);
      }

      const episodeData = await response.json();
      console.log("Loaded episode data:", episodeData);

      // Load the problem
      const problem = await getProblemById(episodeData.problem_id);
      if (!problem) {
        throw new Error(`Problem with ID ${episodeData.problem_id} not found`);
      }

      setProblem(problem);
      setInitCode(episodeData.text);
      setCode(episodeData.text);
      setAttributionLog(episodeData.attribution);
      setInitialTimestep(timestep);
      
      console.log("🎯 Loaded attribution log:", episodeData.attribution.slice(0, 10));
      console.log("🎯 Attribution log length:", episodeData.attribution?.length || 0);
      console.log("🎯 Setting initial timestep to:", timestep);

      return problem;
    } catch (error) {
      console.error("Failed to load episode data:", error);
      setError(error instanceof Error ? error.message : "Failed to load episode data");
      return null;
    } finally {
      setIsProblemLoading(false);
    }
  }, []);

  // Load problem and setup code/attribution from URL parameters
  useEffect(() => {
    const {
      episode,
      timestep,
      problemId,
      startingCode,
      attributionLog: urlAttributionLog,
      maxAssistantActions: urlMaxAssistantActions,
      maxHumanActions: urlMaxHumanActions,
      assistantNoiseProb: urlAssistantNoiseProb,
      assistantNoiseTopK: urlAssistantNoiseTopK,
    } = getZeroStyleParams();
    
    console.log("🔍 ZeroStyleApp URL Parameters:", { episode, timestep, problemId, startingCode, urlAttributionLog });
    
    // Set source episode for API calls
    setSourceEpisode(episode);
    if (urlMaxAssistantActions) {
      setMaxAssistantActions(urlMaxAssistantActions);
    }
    if (urlMaxHumanActions !== undefined) {
      setMaxHumanActions(urlMaxHumanActions);
    }
    if (urlAssistantNoiseProb !== undefined) {
      setAssistantNoiseProb(urlAssistantNoiseProb);
    }
    if (urlAssistantNoiseTopK !== undefined) {
      setAssistantNoiseTopK(urlAssistantNoiseTopK);
    }
    
    const setupProblem = async () => {
      if (episode && timestep) {
        console.log("📁 Loading from episode data:", episode, timestep);
        // Load from episode data
        await loadEpisodeData(episode, timestep);
      } else if (problemId) {
        console.log("🔢 Loading from direct problem ID:", problemId);
        // Load from direct parameters
        const loadedProblem = await loadProblem(problemId);
        if (!loadedProblem) return;

        // Use custom starting code if provided, otherwise use problem's starter code
        const codeToUse = startingCode || loadedProblem.starterCode;
        setInitCode(codeToUse);
        setCode(codeToUse);

        // Use custom attribution log if provided
        if (urlAttributionLog) {
          setAttributionLog(urlAttributionLog);
        }
      } else {
        console.log("❌ No valid parameters found - this will cause random loading");
        setError("Either episode/timestep or problemId is required in the URL parameters");
      }
    };

    setupProblem();
  }, [loadProblem, loadEpisodeData]);

  // Auto-start simulation when editor and problem are ready
  useEffect(() => {
    if (editor && problem && !isProblemLoading && !error) {
      // Wait a short delay to ensure everything is initialized
      const timeout = setTimeout(async () => {
        console.log(`🤖 Auto-starting zero-style simulation for problem: ${problem.title}`);
        
        try {
          await startSimulation({
            type: "state_service_zero_style",
            problemDescription: problem.description,
            intervalMs: 2500, // 2.5 seconds between actions
            maxAssistantActions: Math.max(0, maxAssistantActions),
            humanFollowUpActions: Math.max(0, maxHumanActions),
            assistantNoiseProbability: Math.max(0, Math.min(1, assistantNoiseProb)),
            assistantNoiseTopK: Math.max(1, Math.floor(assistantNoiseTopK)),
            assistantTemperature: 0.75,
            assistantEpsilon: 0.15,
            minActionDelayMs: 350,
            maxActionDelayMs: 900,
            postActionPauseMs: 1400,
            minTypingDelayMs: 45,
            maxTypingDelayMs: 130,
            durationMs: 600000, // 10 minutes max duration
            closeOnStop: true,
          });
        } catch (error) {
          console.error("Failed to start simulation:", error);
        }
      }, 2000); // 2 second delay

      return () => clearTimeout(timeout);
    }
  }, [editor, problem, isProblemLoading, startSimulation, error, maxAssistantActions, maxHumanActions, assistantNoiseProb, assistantNoiseTopK]);

  if (error) {
    return (
      <div className={`app`}>
        <div className="loading-container">
          <h2>❌ Error</h2>
          <p>{error}</p>
          <p>Please check the URL parameters and try again.</p>
        </div>
      </div>
    );
  }

  if (isProblemLoading || !problem) {
    return (
      <div className={`app`}>
        <div className="loading-container">
          <h2>🤖 Loading problem for zero-style simulation...</h2>
        </div>
      </div>
    );
  }

  return (

      <div className={`app`}>
      {showSimulationNotice && (
        <div
          style={overlayStyles.container}
          role="alert"
          aria-live="assertive"
        >
          <div style={overlayStyles.modal}>
            <h2 style={overlayStyles.title}>Zero-style Simulation Running</h2>
            <p style={overlayStyles.body}>
              An automated agent is recording an anchored zero-style episode. Please let the simulation
              finish and avoid typing or interacting with the editor. Close this tab if you need to stop
              the simulation early.
            </p>
          </div>
        </div>
      )}
      <header className="app-header">
        <div className="header-left">
          <h1>[SIMULATION MODE] CODEASSIST - ZERO STYLE</h1>
        </div>
      </header>

      <main className="app-main">
        <ResizablePanels
          initialLeftWidth={UI_CONFIG.PANELS.DEFAULT_LEFT_WIDTH}
          leftPanel={
            <ProblemPanel
              problem={problem}
              onPrev={() => {}}
              onNext={() => {}}
              onRandom={async () => {}} // Disable problem loading in zero-style mode
            />
          }
          rightPanel={
            <VerticalResizablePanels
              showBottomPanel={showTestResults}
              initialTopHeight={500}
              minTopHeight={300}
              maxTopHeight={700}
              topPanel={
                <div className="editor-section">
                  <div className="editor-container">
                    <MonacoEditor
                      initValue={initCode}
                      onChange={setCode}
                      problemId={problem.id}
                      onEditorMount={setEditor}
                      assistantPaused={assistantPaused}
                      onAttributionLogChange={setAttributionLog}
                      zeroStyleConfig={{
                        initialAttributionLog: attributionLog || undefined,
                        initialTurn: initialTimestep,
                        sourceEpisode: sourceEpisode,
                        sourceTimestep: initialTimestep,
                      }}
                    />
                  </div>
                  <EditorFooter
                    submitSolution={submitSolution}
                    loading={submissionState.type === "loading"}
                    assistantPaused={manualAssistantPaused}
                    toggleAssistantPause={() => setManualAssistantPaused(!manualAssistantPaused)}
                  />
                </div>
              }
              bottomPanel={<TestResultsPanel />}
            />
          }
          minLeftWidth={UI_CONFIG.PANELS.MIN_LEFT_WIDTH}
          maxLeftWidth={UI_CONFIG.PANELS.MAX_LEFT_WIDTH}
        />
      </main>
      <Footer />
      <SubmissionModal
        isOpen={isSubmissionModalOpen}
        setIsOpen={setIsSubmissionModalOpen}
        success={submissionState.type === "data" && submissionState.success}
        loadProblemDirect={async () => {}} // Disable problem loading in zero-style mode
      />
    </div>

  );
}

export default ZeroStyleApp;

const overlayStyles = {
  container: {
    position: "fixed" as const,
    inset: 0,
    zIndex: 200,
    background: "rgba(8, 8, 8, 0.18)",
    backdropFilter: "blur(1px)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "24px",
    pointerEvents: "auto" as const,
  },
  modal: {
    maxWidth: "420px",
    width: "100%",
    background: "rgba(18, 18, 18, 0.85)",
    border: "1px solid rgba(255, 255, 255, 0.18)",
    borderRadius: "12px",
    padding: "20px 24px",
    boxShadow: "0px 12px 32px rgba(0, 0, 0, 0.45)",
    textAlign: "center" as const,
    color: "var(--text-primary, #f6f6f6)",
  },
  title: {
    fontSize: "18px",
    marginBottom: "12px",
    letterSpacing: "1.5px",
    textTransform: "uppercase" as const,
    color: "var(--accent, #FFB3BD)",
  },
  body: {
    fontSize: "13px",
    lineHeight: 1.5,
    margin: 0,
  },
};
