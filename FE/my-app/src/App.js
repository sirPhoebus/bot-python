import logo from './logo.svg';
import './App.css';
import React, { useEffect, useState } from 'react';

function App() {
  const [whiteSoldiersData, setWhiteSoldiersData] = useState([]);
  const [russianDollData, setRussianDollData] = useState([]);
  const [bbTrendData, setBbTrendData] = useState([]);
  const [vwapData, setVwapData] = useState([]);
  const [atrData, setAtrData] = useState([]);
  const [obData, setObData] = useState([]);
  const [aggVolData, setAggVolData] = useState([]);
  const [globalData, setGlobalData] = useState([]);

  useEffect(() => {
    const requestOptions = {
      method: 'GET',
      headers: { 
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    };
  
    // fetch('http://localhost:8000/white_soldiers', requestOptions)
    //   .then(response => response.json())
    //   .then(data => setWhiteSoldiersData(data.data));
  
    // fetch('http://localhost:8000/russian_doll', requestOptions)
    //   .then(response => response.json())
    //   .then(data => setRussianDollData(data.data));
  
    fetch('http://localhost:8000/bb_trend?symbol=BTC&tf=1h', requestOptions)
      .then(response => response.json())
      .then(data => setBbTrendData(data.data));
  
    // fetch('http://localhost:8000/vwap', requestOptions)
    //   .then(response => response.json())
    //   .then(data => setVwapData(data.data));
  
    // fetch('http://localhost:8000/atr', requestOptions)
    //   .then(response => response.json())
    //   .then(data => setAtrData(data.data));
  
    // fetch('http://localhost:8000/ob', requestOptions)
    //   .then(response => response.json())
    //   .then(data => setObData(data.data));
  
    // fetch('http://localhost:8000/agg_vol', requestOptions)
    //   .then(response => response.json())
    //   .then(data => setAggVolData(data.data));
  
    // fetch('http://localhost:8000/global', requestOptions)
    //   .then(response => response.json())
    //   .then(data => setGlobalData(data.data));
  }, []);
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        {JSON.stringify(bbTrendData)}
      </header>
    </div>
  );
}

export default App;
