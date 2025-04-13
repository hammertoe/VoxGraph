// static/audio-processor.js

class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);
    // Buffer size can be tuned. Larger buffers = less frequent messages, potentially higher latency.
    // Smaller buffers = more frequent messages, potentially lower latency but more overhead.
    // Let's aim for chunks around 100-250ms. 16000Hz * 0.1s = 1600 samples.
    // Needs to be power of 2 for some operations? Let's use 2048 or 4096.
    this.bufferSize = 4096; // ~250ms at 16kHz
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;

    this.port.onmessage = (event) => {
      // Handle messages from the main thread if needed (e.g., config changes)
      // console.log('Worklet received message:', event.data);
    };
  }

  process(inputs, outputs, parameters) {
    // We expect one input, mono audio
    const input = inputs[0];
    if (!input || !input[0]) {
      // No input data, keep processor alive
      return true;
    }

    const inputChannelData = input[0]; // Assuming mono input

    // Append incoming data to our buffer
    let dataLeft = inputChannelData.length;
    let inputIndex = 0;

    while (dataLeft > 0) {
      const spaceLeft = this.bufferSize - this.bufferIndex;
      const amountToCopy = Math.min(dataLeft, spaceLeft);

      this.buffer.set(inputChannelData.subarray(inputIndex, inputIndex + amountToCopy), this.bufferIndex);

      this.bufferIndex += amountToCopy;
      inputIndex += amountToCopy;
      dataLeft -= amountToCopy;

      // If buffer is full, process and send it
      if (this.bufferIndex >= this.bufferSize) {
        // Convert Float32 to Int16 PCM
        const pcm16Buffer = this.float32ToInt16(this.buffer);

        // Post the Int16Array buffer back to the main thread
        // The second argument is a list of Transferable objects (optional, avoids copying)
        try {
            this.port.postMessage(pcm16Buffer, [pcm16Buffer.buffer]);
        } catch (error) {
            // Firefox might have issues transferring buffers that originated elsewhere.
            // Fallback to copying if transfer fails.
            if (error.name === 'DataCloneError') {
                 console.warn("Transferable failed, copying buffer instead.");
                 this.port.postMessage(this.float32ToInt16(this.buffer)); // Send a copy
            } else {
                 console.error("Error posting message from worklet:", error);
                 // Don't re-throw, try to continue processing
            }
        }


        // Reset buffer index
        this.bufferIndex = 0;
        // Optionally clear the buffer if needed, though overwriting is fine
        // this.buffer.fill(0);
      }
    }

    // Keep the processor alive
    return true;
  }

  // Helper function to convert Float32 [-1.0, 1.0] to Int16 PCM [-32768, 32767]
  float32ToInt16(float32Array) {
    const pcm16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      let val = Math.max(-1, Math.min(1, float32Array[i])); // Clamp to range
      pcm16Array[i] = val * 32767; // Scale to Int16 range
    }
    return pcm16Array;
  }
}

registerProcessor('audio-processor', AudioProcessor);