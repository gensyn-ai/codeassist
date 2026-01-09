import { NextRequest, NextResponse } from 'next/server';
import { readFile } from 'fs/promises';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const episode = searchParams.get('episode');
    const timestep = searchParams.get('timestep');

    if (!episode || !timestep) {
      return NextResponse.json(
        { error: 'Episode and timestep are required' },
        { status: 400 }
      );
    }

    const timestepNum = parseInt(timestep, 10);
    if (isNaN(timestepNum)) {
      return NextResponse.json(
        { error: 'Invalid timestep number' },
        { status: 400 }
      );
    }

    // Path to the episode JSONL file - use Docker container path
    const persistentDataPath = process.env.PERSISTENT_DATA_DIR || '../persistent-data';
    const episodePath = `${persistentDataPath}/state-service/episodes/${episode}/raw/${episode}.jsonl`;

    console.log('Looking for episode file at:', episodePath);

    try {
      // Read the file
      const fileContent = await readFile(episodePath, 'utf-8');
      const lines = fileContent.split('\n').filter(line => line.trim());

      if (lines.length === 0) {
        return NextResponse.json(
          { error: 'Empty episode file' },
          { status: 404 }
        );
      }

      // Parse first line to get problem_id
      const firstLine = JSON.parse(lines[0]);
      const problemId = firstLine.problem_id;

      if (!problemId) {
        return NextResponse.json(
          { error: 'No problem_id found in episode header' },
          { status: 400 }
        );
      }

      // Get the specific timestep line
      // File structure: Line 0 = Header, Line 1 = timestep 0, Line 2 = timestep 1, etc.
      // So timestep N is at line index N+1
      const lineIndex = timestepNum + 1;
      
      if (lineIndex >= lines.length) {
        return NextResponse.json(
          { error: `Timestep ${timestepNum} exceeds available data (max timestep: ${lines.length - 2})` },
          { status: 400 }
        );
      }

      const timestepLine = lines[lineIndex];
      const timestepData = JSON.parse(timestepLine);

      const text = timestepData.text || '';
      const attribution = timestepData.attribution || [];

      if (!text) {
        return NextResponse.json(
          { error: 'No text found at specified timestep' },
          { status: 400 }
        );
      }

      return NextResponse.json({
        problem_id: problemId,
        text: text,
        attribution: attribution
      });

    } catch (fileError) {
      console.error('Error reading episode file:', fileError);
      return NextResponse.json(
        { error: 'Failed to read episode file' },
        { status: 500 }
      );
    }

  } catch (error) {
    console.error('Error in episode-data API:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
