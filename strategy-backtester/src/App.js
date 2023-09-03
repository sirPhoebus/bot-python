// src/App.js

import React, { useState } from 'react';  // <-- Make sure to import useState
import './App.css';
import StrategyBacktester from './StrategyBacktester';


function App() {
  const [data, setData] = useState([]);
  return (
    <div className="App">
       <StrategyBacktester />
      
    </div>
  );
}

export default App;
