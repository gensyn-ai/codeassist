import * as monaco from "monaco-editor";
import type { HumanSimulator, SimulationAction, SimulationConfig, SimulationStats } from "../types";
import { InferenceRequest, stateServiceClient } from "../../services/stateService";
import { initialAttributionLog, AttributionLog } from "../../attribution/log";

/**
 * StateServiceHumanSimulator - A human simulator that generates code using the state service's inference_human endpoint
 * 
 * This simulator calls the /inference_human endpoint which uses a human policy model to determine
 * what actions to take and how to modify the code. Instead of applying unified diffs, it manually
 * applies actions based on the ActionIndex returned by the model.
 * 
 * Action Types:
 * 0 (NO_OP): Do nothing
 * 1 (FILL_PARTIAL_LINE): Complete the line from cursor position
 * 2 (REPLACE_AND_APPEND_SINGLE_LINE): Replace current line content (preserve indentation)
 * 3 (REPLACE_AND_APPEND_MULTI_LINE): Replace current line content (preserve indentation)
 * 4 (EDIT_EXISTING_LINES): Replace line at target_line with response text
 * 5 (EXPLAIN_SINGLE_LINES): Add "#" comment at end of current line
 * 6 (EXPLAIN_MULTI_LINE): Add "#" comment at target_line
 * 
 * Prerequisites:
 * 1. State service must be running and accessible
 * 2. The state service must have the inference_human endpoint available
 * 3. Human policy model must be trained and available in the policy_models service
 */

export class StateServiceHumanSimulator implements HumanSimulator {
  private editor: monaco.editor.IStandaloneCodeEditor | null = null;
  private config: SimulationConfig;
  private stats: SimulationStats = {
    totalActions: 0,
    totalDurationMs: 0,
    episodesCreated: 0,
    agentSuggestionsReceived: 0,
  };
  private isRunning = false;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private startTime: number = 0;
  private isProcessing = false; // Mutex to prevent concurrent processing
  private isTyping = false; // Mutex to prevent concurrent typing operations
  private problemDescription?: string;
  private timestep = 0;
  private attributionLog: AttributionLog = initialAttributionLog("");
  private closeOnStop = false;
  private maxAssistantActions?: number;
  private humanFollowUpActions: number = 0;
  private assistantActionsPerformed = 0;
  private humanActionsPerformed = 0;
  private assistantNoiseProbability = 0;
  private assistantNoiseTopK = 3;
  private assistantTemperature: number | null = null;
  private assistantEpsilon: number | null = null;
  private minActionDelayMs = 350;
  private maxActionDelayMs = 900;
  private postActionPauseMs = 1400;
  private minTypingDelayMs = 45;
  private maxTypingDelayMs = 130;

  constructor(config: SimulationConfig) {
    this.config = config;
    this.problemDescription = config.problemDescription;
    this.closeOnStop = !!config.closeOnStop;
    if (config.maxAssistantActions !== undefined) {
      this.maxAssistantActions = Math.max(0, config.maxAssistantActions);
      const provided = config.humanFollowUpActions;
      this.humanFollowUpActions = Math.max(1, provided !== undefined ? provided : 1);
    } else if (config.humanFollowUpActions !== undefined) {
      this.humanFollowUpActions = Math.max(0, config.humanFollowUpActions);
    } else if (config.maxActions !== undefined) {
      this.humanFollowUpActions = Math.max(0, config.maxActions);
    }

    if (typeof config.assistantNoiseProbability === "number" && !Number.isNaN(config.assistantNoiseProbability)) {
      this.assistantNoiseProbability = Math.min(Math.max(config.assistantNoiseProbability, 0), 1);
    }

    if (typeof config.assistantNoiseTopK === "number" && !Number.isNaN(config.assistantNoiseTopK)) {
      const candidate = Math.floor(config.assistantNoiseTopK);
      if (candidate > 0) {
        this.assistantNoiseTopK = Math.min(candidate, 7);
      }
    }

    if (typeof config.assistantTemperature === "number" && config.assistantTemperature > 0) {
      this.assistantTemperature = config.assistantTemperature;
    }

    if (typeof config.assistantEpsilon === "number" && config.assistantEpsilon >= 0) {
      this.assistantEpsilon = config.assistantEpsilon;
    }

    if (typeof config.minActionDelayMs === "number" && config.minActionDelayMs >= 0) {
      this.minActionDelayMs = config.minActionDelayMs;
    }
    if (typeof config.maxActionDelayMs === "number" && config.maxActionDelayMs >= this.minActionDelayMs) {
      this.maxActionDelayMs = config.maxActionDelayMs;
    }
    if (typeof config.postActionPauseMs === "number" && config.postActionPauseMs >= 0) {
      this.postActionPauseMs = config.postActionPauseMs;
    }
    if (typeof config.minTypingDelayMs === "number" && config.minTypingDelayMs >= 0) {
      this.minTypingDelayMs = config.minTypingDelayMs;
    }
    if (typeof config.maxTypingDelayMs === "number" && config.maxTypingDelayMs >= this.minTypingDelayMs) {
      this.maxTypingDelayMs = config.maxTypingDelayMs;
    }
  }

  setEditor(editor: monaco.editor.IStandaloneCodeEditor | null) {
    this.editor = editor;
  }

  setAttributionLog(attributionLog: AttributionLog): void {
    console.log("🎯 StateServiceHumanSimulator received attribution log:", attributionLog);
    console.log("🎯 Attribution log length:", attributionLog?.length || 0);
    this.attributionLog = attributionLog;
  }

  setInitialTimestep(timestep: number): void {
    console.log("🎯 Setting initial timestep to:", timestep);
    // The provided timestep represents the last completed state in the attribution log.
    // The next inference should start from timestep + 1.
    this.timestep = timestep + 1;
    console.log("🎯 Next inference will start at timestep:", this.timestep);
  }

  getStats(): SimulationStats {
    return { ...this.stats };
  }

  async start(): Promise<void> {
    if (this.isRunning || !this.editor) {
      return;
    }

    this.isRunning = true;
    this.startTime = Date.now();
    this.stats.episodesCreated += 1;
    // Don't reset timestep here - preserve the initial timestep set by setInitialTimestep
    
    console.log("Starting State Service Human Simulator...");

    // First, simulate clicking on the editor to focus it
    await this.simulateEditorClick();

    if (this.maxAssistantActions && this.maxAssistantActions > 0) {
      this.startZeroStyleSequence();
    } else {
      // Start the periodic human simulation
      this.startPeriodicHumanSimulation();
    }
  }

  async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;
    this.isProcessing = false; // Reset processing state when stopping
    
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    this.stats.totalDurationMs += Date.now() - this.startTime;
    this.stats.lastRunTime = Date.now();
    
    console.log("Stopped State Service Human Simulator");

    const completion = new Promise<void>((resolve) => {
      const HANDSHAKE_TIMEOUT_MS = 2000;
      let resolved = false;
      let timeoutId: ReturnType<typeof setTimeout> | null = null;

      const resolveAndClear = () => {
        if (!resolved) {
          resolved = true;
          if (timeoutId !== null) {
            clearTimeout(timeoutId);
          }
          resolve();
        }
      };

      const fallback = () => {
        if (!resolved) {
          console.warn(
            "zeroStyleSimulationStopping handshake timed out; proceeding without confirmation",
          );
          resolveAndClear();
        }
      };
      try {
        const event = new CustomEvent<{ resolve: () => void }>(
          "zeroStyleSimulationStopping",
          {
            detail: {
              resolve: resolveAndClear,
            },
          },
        );
        window.dispatchEvent(event);
        // Ensure completion even if no listener responds (or never calls resolve)
        timeoutId = window.setTimeout(fallback, HANDSHAKE_TIMEOUT_MS);
      } catch (err) {
        console.warn("Failed to dispatch zeroStyleSimulationStopping", err);
        resolveAndClear();
      }
    });

    await completion;

    if (this.closeOnStop) {
      try {
        window.close();
      } catch (err) {
        console.warn("Unable to close window automatically", err);
        window.location.href = "about:blank";
      }
    }
  }

  private startPeriodicHumanSimulation() {
    const intervalMs = this.config.intervalMs || 3500;
    const maxActions = this.config.maxActions || 50;
    const duration = this.config.durationMs || 300000; // Default 5 minutes

    let actionCount = 0;

    this.intervalId = setInterval(async () => {
      if (!this.isRunning || !this.editor) {
        this.stop();
        return;
      }

      // Check if we've reached max actions or duration
      if (actionCount >= maxActions || (Date.now() - this.startTime) >= duration) {
        this.stop();
        return;
      }

      // Generate next action using state service human model
      const action: SimulationAction = {
        type: "state_service_human",
        timestamp: Date.now(),
      };

      await this.executeAction(action);
      actionCount++;
    }, intervalMs);
  }

  private startZeroStyleSequence() {
    const intervalMs = this.config.intervalMs || 2500;
    const assistantLimit = this.maxAssistantActions ?? 0;
    const followUpHumans = Math.max(1, this.humanFollowUpActions ?? 1);

    const runSequence = async () => {
      try {
        for (let i = 0; i < assistantLimit && this.isRunning; i += 1) {
          await this.executeAction({ type: "state_service_human", timestamp: Date.now() });
          this.humanActionsPerformed += 1;
          if (!this.isRunning) break;
          if (intervalMs > 0) {
            await this.simulateWait(intervalMs);
          }
          if (!this.isRunning) break;
          await this.executeAction({ type: "state_service_assistant", timestamp: Date.now() });
          this.assistantActionsPerformed += 1;
          if (!this.isRunning) break;
          if (intervalMs > 0 && (i < assistantLimit - 1 || followUpHumans > 0)) {
            await this.simulateWait(intervalMs);
          }
        }

        if (this.isRunning) {
          for (let j = 0; j < followUpHumans && this.isRunning; j += 1) {
            await this.executeAction({ type: "state_service_human", timestamp: Date.now() });
            this.humanActionsPerformed += 1;
            if (!this.isRunning) break;
            if (intervalMs > 0 && j < followUpHumans - 1) {
              await this.simulateWait(intervalMs);
            }
          }
        }
      } catch (error) {
        console.error("Zero-style sequence failed:", error);
      } finally {
        if (this.isRunning) {
          await this.stop();
        }
      }
    };

    void runSequence();
  }

  async executeAction(action: SimulationAction): Promise<void> {
    if (!this.editor) {
      return;
    }

    switch (action.type) {
      case "state_service_human":
        await this.generateAndApplyHumanAction();
        break;
      case "state_service_assistant":
        await this.generateAndApplyAssistantAction();
        break;
      case "type":
        if (action.payload?.text) {
          await this.simulateTyping(action.payload.text);
        }
        break;
      case "cursor_move":
        if (action.payload?.position) {
          await this.simulateCursorMove(action.payload.position);
        }
        break;
      case "wait":
        if (action.payload?.durationMs) {
          await this.simulateWait(action.payload.durationMs);
        }
        break;
    }

    this.stats.totalActions += 1;
  }

  private async generateAndApplyHumanAction(): Promise<void> {
    if (!this.editor) return;

    // Check if already processing - skip if so
    if (this.isProcessing) {
      console.log("🔒 Skipping generation - already processing");
      return;
    }

    try {
      // Set processing lock
      this.isProcessing = true;

      await this.humanlikeDelay();

      // Get current context from the editor
      const model = this.editor.getModel();
      if (!model) return;

      const position = this.editor.getPosition();
      if (!position) return;

      // Get full file content
      let fullFileContent = model.getValue();
      let adjustedPosition = position;
      let problemBlockOffset = 0;
      let problemBlockLines = 0;

      // Include problem description as a comment block if available (like Ollama simulator)
      if (this.problemDescription && !fullFileContent.includes(this.problemDescription)) {
        const problemBlock = `'''\nProblem: ${this.problemDescription}\n'''\n`;
        fullFileContent = problemBlock + fullFileContent;
        
        // Calculate the offset caused by adding the problem block
        problemBlockOffset = problemBlock.length;
        problemBlockLines = problemBlock.split('\n').length - 1; // -1 because split adds extra empty element
        
        // Adjust cursor position to account for the added lines
        adjustedPosition = new monaco.Position(
          position.lineNumber + problemBlockLines,
          position.column
        );
      }

      // Use the stored attribution log
      const attributionLog = this.attributionLog;
      
      console.log("🎯 Using attribution log in inference request:", attributionLog);
      console.log("🎯 Attribution log length in request:", attributionLog?.length || 0);

      // Calculate the adjusted cursor offset
      const adjustedCursorOffset = model.getOffsetAt(position) + problemBlockOffset;

      // Create inference request for the human model
      const currentTimestep = this.timestep++;
      console.log("🎯 Current timestep being sent to inference:", currentTimestep);
      const inferenceRequest: InferenceRequest = {
        text: fullFileContent,
        author_attribution: "user",
        timestep: currentTimestep,
        timestamp: new Date().toISOString(),
        context: {
          file_type: "python", // Could be made configurable
          cursor_position: {
            line: adjustedPosition.lineNumber,
            column: adjustedPosition.column,
          },
          selection: null, // No selection for simulation
          file_length: fullFileContent.length,
          cursorOffset: adjustedCursorOffset,
        },
        attribution: attributionLog,
      };
      
      // Call the inference_human endpoint using the state service client
      const response = await stateServiceClient.generateInferenceHuman(inferenceRequest);
      
      if (response && response.metadata && response.metadata.action) {
        const [actionIndex, targetLine] = response.metadata.action;
        const responseText = response.response || "";
        
        console.log(`🧠 State service generated action ${actionIndex} (target line: ${targetLine}): "${responseText.substring(0, 50)}..."`);
        
        // Adjust target line if we added a problem block
        let adjustedTargetLine = targetLine;
        if (problemBlockOffset > 0) {
          adjustedTargetLine = Math.max(1, targetLine - problemBlockLines);
        }
        
        // Dispatch event with predicted human action data
        const event = new CustomEvent('predictedHumanAction', {
          detail: {
            actionIndex,
            lineNumber: adjustedTargetLine,
            timestamp: Date.now()
          }
        });
        window.dispatchEvent(event);
        
        // Apply the action based on the ActionIndex (use original position for editor operations)
        await this.applyAction(actionIndex, adjustedTargetLine, responseText, position);
        
      } else {
        // Fallback if no action was generated
        if (!this.isTyping) {
          this.isTyping = true;
          try {
            await this.simulateTypingWithDelay('\n');
          } finally {
            this.isTyping = false;
          }
        }
      }

    } catch (error) {
      console.error('Error calling state service human inference:', error);
      // Fallback to simple typing
      if (!this.isTyping) {
        this.isTyping = true;
        try {
          await this.simulateTypingWithDelay('\n');
        } finally {
          this.isTyping = false;
        }
      }
    } finally {
      // Always release the processing lock
      this.isProcessing = false;
      await this.simulateWait(this.postActionPauseMs);
    }
  }

  private async generateAndApplyAssistantAction(): Promise<void> {
    if (!this.editor) return;

    if (this.isProcessing) {
      console.log("🔒 Skipping assistant generation - already processing");
      return;
    }

    try {
      this.isProcessing = true;

      await this.humanlikeDelay();

      const model = this.editor.getModel();
      if (!model) return;

      const position = this.editor.getPosition();
      if (!position) return;

      let fullFileContent = model.getValue();
      let adjustedPosition = position;
      let problemBlockOffset = 0;
      let problemBlockLines = 0;

      if (this.problemDescription && !fullFileContent.includes(this.problemDescription)) {
        const problemBlock = `'''\nProblem: ${this.problemDescription}\n'''\n`;
        fullFileContent = problemBlock + fullFileContent;
        problemBlockOffset = problemBlock.length;
        problemBlockLines = problemBlock.split('\n').length - 1;
        adjustedPosition = new monaco.Position(
          position.lineNumber + problemBlockLines,
          position.column,
        );
      }

      const adjustedCursorOffset = model.getOffsetAt(position) + problemBlockOffset;

      const currentTimestep = this.timestep++;

      const explorationEnabled = this.shouldInjectAssistantNoise();
      if (explorationEnabled) {
        console.log(
          `🎲 Injecting assistant noise by sampling assistant policy top-k (k=${this.assistantNoiseTopK})`,
        );
      }

      const inferenceContext: InferenceRequest["context"] = {
        file_type: "python",
        cursor_position: {
          line: adjustedPosition.lineNumber,
          column: adjustedPosition.column,
        },
        selection: null,
        file_length: fullFileContent.length,
        cursorOffset: adjustedCursorOffset,
      };

      if (explorationEnabled) {
        const topK = Math.max(1, Math.min(this.assistantNoiseTopK, 7));
        inferenceContext.assistantStrategyOverride = "sample_top_k";
        inferenceContext.assistantTopK = topK;
        if (this.assistantTemperature && this.assistantTemperature > 0) {
          inferenceContext.assistantTemperature = this.assistantTemperature;
        }
        if (this.assistantEpsilon !== null && this.assistantEpsilon >= 0) {
          inferenceContext.assistantEpsilon = this.assistantEpsilon;
        }
      } else {
        if (this.assistantTemperature && this.assistantTemperature > 0) {
          inferenceContext.assistantTemperature = this.assistantTemperature;
        }
        if (this.assistantEpsilon !== null && this.assistantEpsilon >= 0) {
          inferenceContext.assistantEpsilon = this.assistantEpsilon;
        }
      }

      const inferenceRequest: InferenceRequest = {
        text: fullFileContent,
        author_attribution: "assistant",
        timestep: currentTimestep,
        timestamp: new Date().toISOString(),
        context: inferenceContext,
        attribution: this.attributionLog,
      };

      const response = await stateServiceClient.generateInference(inferenceRequest);

      if (response && response.metadata && response.metadata.action) {
        const [actionIndex, targetLine] = response.metadata.action;
        const responseText = response.response || "";

        let adjustedTargetLine = targetLine;
        if (problemBlockOffset > 0) {
          adjustedTargetLine = Math.max(1, targetLine - problemBlockLines);
        }

        await this.applyAction(actionIndex, adjustedTargetLine, responseText, position);
      } else {
        console.warn("Assistant inference returned no action; falling back to NO-OP");
      }
    } catch (error) {
      console.error('Error calling state service assistant inference:', error);
    } finally {
      this.isProcessing = false;
      await this.simulateWait(this.postActionPauseMs);
    }
  }

  private shouldInjectAssistantNoise(): boolean {
    return this.assistantNoiseProbability > 0 && Math.random() < this.assistantNoiseProbability;
  }

  private randomDelay(min: number, max: number): number {
    if (max <= min) return min;
    return min + Math.random() * (max - min);
  }

  private async humanlikeDelay(min = this.minActionDelayMs, max = this.maxActionDelayMs): Promise<void> {
    const delay = this.randomDelay(min, max);
    await this.simulateWait(delay);
  }

  private async applyAction(
    actionIndex: number, 
    targetLine: number, 
    responseText: string, 
    currentPosition: monaco.Position
  ): Promise<void> {
    if (!this.editor) return;

    const model = this.editor.getModel();
    if (!model) return;

    try {
      switch (actionIndex) {
        case 0: // NO_OP
          console.log('🔄 NO_OP action - doing nothing');
          break;

        case 1: // FILL_PARTIAL_LINE
          await this.fillPartialLine(responseText, currentPosition);
          break;

        case 2: // REPLACE_AND_APPEND_SINGLE_LINE
        case 3: // REPLACE_AND_APPEND_MULTI_LINE
          await this.replaceAndAppendLine(responseText, currentPosition);
          break;

        case 4: // EDIT_EXISTING_LINES
          await this.editExistingLines(responseText, targetLine);
          break;

        case 5: // EXPLAIN_SINGLE_LINES
          await this.addInlineComment(responseText, currentPosition);
          break;

        case 6: // EXPLAIN_MULTI_LINE
          await this.addCommentAtLine(responseText, targetLine);
          break;

        default:
          console.log(`⚠️ Unknown action index: ${actionIndex}, falling back to typing`);
          if (!this.isTyping) {
            this.isTyping = true;
            try {
              await this.simulateTypingWithDelay(responseText);
            } finally {
              this.isTyping = false;
            }
          }
          break;
      }
    } catch (error) {
      console.error(`Error applying action ${actionIndex}:`, error);
      // Fallback to simple typing
      if (!this.isTyping) {
        this.isTyping = true;
        try {
          await this.simulateTypingWithDelay('\n');
        } finally {
          this.isTyping = false;
        }
      }
    }
  }

  // Helper method to get indentation level of a line
  private getIndentationLevel(line: string): number {
    return line.length - line.trimStart().length;
  }

  // Helper method to get indentation string from a line
  private getIndentation(line: string): string {
    const indentLevel = this.getIndentationLevel(line);
    return line.substring(0, indentLevel);
  }

  // Action 1: FILL_PARTIAL_LINE - Complete the line from cursor position
  private async fillPartialLine(responseText: string, position: monaco.Position): Promise<void> {
    if (!this.editor || this.isTyping) return;

    // Set typing mutex
    this.isTyping = true;

    try {
      // Clean response text to avoid newlines for single line completion
      const cleanText = responseText.replace(/\n/g, ' ').trim();
      
      // Set cursor to the correct position first
      this.editor.setPosition(position);
      
      // Type with delay for realistic simulation
      await this.simulateTypingWithDelay(cleanText);
      
      console.log(`📝 Filled partial line: "${cleanText}"`);
    } finally {
      this.isTyping = false;
    }
  }

  // Actions 2 & 3: REPLACE_AND_APPEND - Replace current line content (preserve indentation)
  private async replaceAndAppendLine(responseText: string, position: monaco.Position): Promise<void> {
    if (!this.editor || !this.editor.getModel() || this.isTyping) return;

    // Set typing mutex
    this.isTyping = true;

    try {
      const model = this.editor.getModel()!;
      const currentLine = model.getLineContent(position.lineNumber);
      const indentation = this.getIndentation(currentLine);
      
      // Clear the current line content (except indentation)
      const lineRange = new monaco.Range(
        position.lineNumber, indentation.length + 1, 
        position.lineNumber, currentLine.length + 1
      );

      this.editor.executeEdits("simulation-clear-line", [{
        range: lineRange,
        text: "",
      }]);

      // Set cursor to start of line content (after indentation)
      const startPosition = new monaco.Position(position.lineNumber, indentation.length + 1);
      this.editor.setPosition(startPosition);
      
      // Clean and prepare the response text
      const lines = responseText.split('\n');
      const formattedText = lines
        .map((line, index) => {
          // First line doesn't need indentation (cursor is already after indentation)
          if (index === 0) {
            return line.trim();
          } else {
            return indentation + line; // Keep relative indentation for multi-line
          }
        })
        .join('\n');

      // Type with delay for realistic simulation
      await this.simulateTypingWithDelay(formattedText);
      
      console.log(`🔄 Replaced and appended line: "${formattedText.substring(0, 50)}..."`);
    } finally {
      this.isTyping = false;
    }
  }

  // Action 4: EDIT_EXISTING_LINES - Replace line at target_line with response text
  private async editExistingLines(responseText: string, targetLine: number): Promise<void> {
    if (!this.editor || !this.editor.getModel() || this.isTyping) return;

    const model = this.editor.getModel()!;
    
    // Ensure target line is within bounds
    if (targetLine < 1 || targetLine > model.getLineCount()) {
      console.log(`⚠️ Target line ${targetLine} out of bounds, using current position`);
      const position = this.editor.getPosition();
      if (position) {
        await this.replaceAndAppendLine(responseText, position);
      }
      return;
    }

    // Set typing mutex
    this.isTyping = true;

    try {
      const currentLine = model.getLineContent(targetLine);
      const indentation = this.getIndentation(currentLine);
      
      // Clear the target line content (except indentation)
      const lineRange = new monaco.Range(
        targetLine, indentation.length + 1, 
        targetLine, currentLine.length + 1
      );

      this.editor.executeEdits("simulation-clear-target-line", [{
        range: lineRange,
        text: "",
      }]);

      // Set cursor to start of line content (after indentation)
      const startPosition = new monaco.Position(targetLine, indentation.length + 1);
      this.editor.setPosition(startPosition);
      
      // Format response with proper indentation
      const lines = responseText.split('\n');
      const formattedText = lines
        .map((line, index) => {
          // First line doesn't need indentation (cursor is already after indentation)
          if (index === 0) {
            return line.trim();
          } else {
            return indentation + line;
          }
        })
        .join('\n');

      // Type with delay for realistic simulation
      await this.simulateTypingWithDelay(formattedText);
      
      // Move cursor back to end of file
      this.moveCursorToEndOfFile();

    } finally {
      this.isTyping = false;
    }
  }

  // Action 5: EXPLAIN_SINGLE_LINES - Add "#" comment at end of current line
  private async addInlineComment(responseText: string, position: monaco.Position): Promise<void> {
    if (!this.editor || !this.editor.getModel() || this.isTyping) return;

    // Set typing mutex
    this.isTyping = true;

    try {
      const model = this.editor.getModel()!;
      const currentLine = model.getLineContent(position.lineNumber);
      
      // Clean response text for inline comment
      const cleanComment = responseText.replace(/\n/g, ' ').trim();
      const commentText = currentLine.trim() === '' ? `# ${cleanComment}` : ` # ${cleanComment}`;
      
      // Set cursor to end of line
      const lineEndPosition = new monaco.Position(position.lineNumber, currentLine.length + 1);
      this.editor.setPosition(lineEndPosition);
      
      // Type with delay for realistic simulation
      await this.simulateTypingWithDelay(commentText);

    } finally {
      this.isTyping = false;
    }
  }

  // Action 6: EXPLAIN_MULTI_LINE - Add "#" comment at target_line
  private async addCommentAtLine(responseText: string, targetLine: number): Promise<void> {
    if (!this.editor || !this.editor.getModel() || this.isTyping) return;

    const model = this.editor.getModel()!;
    
    // Ensure target line is within bounds
    if (targetLine < 1 || targetLine > model.getLineCount()) {
      console.log(`⚠️ Target line ${targetLine} out of bounds, using current position`);
      const position = this.editor.getPosition();
      if (position) {
        await this.addInlineComment(responseText, position);
      }
      return;
    }

    // Set typing mutex
    this.isTyping = true;

    try {
      const targetLineContent = model.getLineContent(targetLine);
      const indentation = this.getIndentation(targetLineContent);
      
      // Set cursor to beginning of target line
      const insertPosition = new monaco.Position(targetLine, 1);
      this.editor.setPosition(insertPosition);
      
      // Format comment with proper indentation
      const commentLines = responseText.split('\n');
      const formattedComment = commentLines
        .map(line => `${indentation}# ${line.trim()}`)
        .join('\n') + '\n';
      
      // Type with delay for realistic simulation
      await this.simulateTypingWithDelay(formattedComment);
      
      // Move cursor back to end of file
      this.moveCursorToEndOfFile();

    } finally {
      this.isTyping = false;
    }
  }

  // Helper method to move cursor to the end of the file
  private moveCursorToEndOfFile(): void {
    if (!this.editor) return;
    
    const model = this.editor.getModel();
    if (model) {
      const lastLineNumber = model.getLineCount();
      const lastLineLength = model.getLineContent(lastLineNumber).length;
      const position = new monaco.Position(lastLineNumber, lastLineLength + 1);
      this.editor.setPosition(position);
    }
  }

  private async simulateEditorClick(): Promise<void> {
    if (!this.editor) return;
    
    // Focus the editor (simulates clicking on it)
    this.editor.focus();
    
    // Set cursor to after the last line of the editor
    const model = this.editor.getModel();
    if (model) {
      const lastLineNumber = model.getLineCount();
      const lastLineLength = model.getLineContent(lastLineNumber).length;
      const position = new monaco.Position(lastLineNumber, lastLineLength + 1);
      this.editor.setPosition(position);
    }
    
    console.log("Simulated click on editor - focused and positioned cursor after last line");
  }

  private async simulateTyping(text: string): Promise<void> {
    if (!this.editor) return;
    
    const position = this.editor.getPosition();
    if (!position) return;

    const edit = {
      range: new monaco.Range(position.lineNumber, position.column, position.lineNumber, position.column),
      text: text,
    };

    this.editor.executeEdits("simulation-state-service-type", [edit]);
    
    // Move cursor after the inserted text
    const lines = text.split('\n');
    let newLineNumber = position.lineNumber;
    let newColumn = position.column;
    
    if (lines.length > 1) {
      newLineNumber += lines.length - 1;
      newColumn = lines[lines.length - 1].length + 1;
    } else {
      newColumn += text.length;
    }
    
    const newPosition = new monaco.Position(newLineNumber, newColumn);
    this.editor.setPosition(newPosition);

    console.log(`Simulated typing: "${text.replace(/\n/g, '\\n')}"`);
  }

  private async simulateTypingWithDelay(text: string): Promise<void> {
    if (!this.editor) return;
    
    // The isTyping mutex should be handled by the caller
    // This method should only be called when it's safe to type
    
    for (const token of text.split('')) {
      this.editor.trigger('simulation', 'type', { text: token });
      const delay = this.randomDelay(this.minTypingDelayMs, this.maxTypingDelayMs);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
    
    console.log(`Simulated typing with delay: "${text.replace(/\n/g, '\\n')}"`);
  }

  private async simulateCursorMove(position: { line: number; column: number }): Promise<void> {
    if (!this.editor) return;
    
    const newPosition = new monaco.Position(position.line, position.column);
    this.editor.setPosition(newPosition);
    console.log(`Simulated cursor move to line ${position.line}, column ${position.column}`);
  }

  private async simulateWait(durationMs: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, durationMs));
  }
}
