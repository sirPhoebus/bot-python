import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [word, setWord] = useState('');

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:8765/');

    socket.onopen = (event) => {
      console.log('Connected to the WebSocket server', event);
    };

    socket.onmessage = (event) => {
      setWord(event.data);
    };

    socket.onerror = (error) => {
      console.error('WebSocket Error:', error);
    };

    socket.onclose = (event) => {
      if (event.wasClean) {
        console.log(`Closed cleanly, code=${event.code}, reason=${event.reason}`);
      } else {
        console.warn('Connection died');
      }
    };

    // Cleanup the WebSocket connection when the component is unmounted
    return () => {
      socket.close();
    };
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>{word.toUpperCase()}</h1>
      </header>
    </div>
  );
}

export default App;
