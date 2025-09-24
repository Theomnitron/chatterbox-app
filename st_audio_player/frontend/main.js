// This file is the frontend for your custom Streamlit component.
// It will be compiled into a single JavaScript file.
import React, { useEffect, useRef, useState } from "react";
import {
  Streamlit,
  useRenderData,
  withStreamlitConnection,
} from "streamlit-component-lib";

const AudioPlayer = (props) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const audioContextRef = useRef(null);
  const audioQueueRef = useRef([]);
  const sourceNodeRef = useRef(null);
  const renderData = useRenderData();

  useEffect(() => {
    // Initialize AudioContext
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
  }, []);

  useEffect(() => {
    if (renderData && renderData.args && renderData.args.data) {
      const { data, stream_active } = renderData.args;

      if (data) {
        // Decode the incoming audio chunk and add to the queue
        const buffer = Uint8Array.from(data.match(/.{2}/g).map(byte => parseInt(byte, 16))).buffer;
        audioContextRef.current.decodeAudioData(buffer, (decodedBuffer) => {
          audioQueueRef.current.push(decodedBuffer);
          if (!isPlaying && audioQueueRef.current.length > 0) {
            playNextChunk();
          }
        });
        
        setIsReady(true);
      }
      
      // If the stream is no longer active, we're done receiving chunks
      if (!stream_active) {
          setIsReady(false);
          // Clean up when done
          console.log("Stream complete.");
      }
    }
  }, [renderData]);

  const playNextChunk = () => {
    if (audioQueueRef.current.length > 0 && audioContextRef.current && !sourceNodeRef.current) {
      const nextChunk = audioQueueRef.current.shift();
      const source = audioContextRef.current.createBufferSource();
      source.buffer = nextChunk;
      source.connect(audioContextRef.current.destination);
      source.start();
      
      source.onended = () => {
        source.disconnect();
        sourceNodeRef.current = null;
        if (audioQueueRef.current.length > 0) {
          playNextChunk();
        } else if (!isReady) {
            // All chunks played, reset
            setIsPlaying(false);
        }
      };
      
      sourceNodeRef.current = source;
      setIsPlaying(true);
    }
  };

  const handlePlayPause = () => {
    if (!isPlaying) {
      if (audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume();
      }
      playNextChunk();
    } else {
      if (sourceNodeRef.current) {
          sourceNodeRef.current.stop();
          sourceNodeRef.current = null;
          setIsPlaying(false);
      }
    }
  };

  useEffect(() => {
      Streamlit.setComponentReady();
  }, []);

  return (
    <div style={{ padding: "1rem", border: "1px solid #ccc", borderRadius: "8px" }}>
      <h4>Streaming Player</h4>
      <button 
        onClick={handlePlayPause} 
        disabled={!isReady && audioQueueRef.current.length === 0}
        style={{ padding: "0.5rem 1rem", fontSize: "1rem", cursor: "pointer" }}
      >
        {isPlaying ? "⏸️ Pause" : "▶️ Play"}
      </button>
      <div style={{ marginTop: "1rem" }}>
        {isReady ? "Streaming..." : "Ready."}
      </div>
    </div>
  );
};

export default withStreamlitConnection(AudioPlayer);
