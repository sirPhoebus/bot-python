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
        'symbol': 'BTC',
        'tf': '1h'
      }
    };
  
    fetch('http://localhost:8000/white_soldiers', requestOptions)
      .then(response => response.json())
      .then(data => setWhiteSoldiersData(data.data));
  
    fetch('http://localhost:8000/russian_doll', requestOptions)
      .then(response => response.json())
      .then(data => setRussianDollData(data.data));
  
    fetch('http://localhost:8000/bb_trend', requestOptions)
      .then(response => response.json())
      .then(data => setBbTrendData(data.data));
  
    fetch('http://localhost:8000/vwap', requestOptions)
      .then(response => response.json())
      .then(data => setVwapData(data.data));
  
    fetch('http://localhost:8000/atr', requestOptions)
      .then(response => response.json())
      .then(data => setAtrData(data.data));
  
    fetch('http://localhost:8000/ob', requestOptions)
      .then(response => response.json())
      .then(data => setObData(data.data));
  
    fetch('http://localhost:8000/agg_vol', requestOptions)
      .then(response => response.json())
      .then(data => setAggVolData(data.data));
  
    fetch('http://localhost:8000/global', requestOptions)
      .then(response => response.json())
      .then(data => setGlobalData(data.data));
  }, []);

  return (
    <div>
      <h1>FastAPI Dashboard</h1>
      <table>
        <thead>
          <tr>
            <th>Endpoint</th>
            <th>Data</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>white_soldiers</td>
            <td>{JSON.stringify(whiteSoldiersData)}</td>
          </tr>
          <tr>
            <td>russian_doll</td>
            <td>{JSON.stringify(russianDollData)}</td>
          </tr>
          <tr>
            <td>bb_trend</td>
            <td>{JSON.stringify(bbTrendData)}</td>
          </tr>
          <tr>
            <td>vwap</td>
            <td>{JSON.stringify(vwapData)}</td>
          </tr>
          <tr>
            <td>atr</td>
            <td>{JSON.stringify(atrData)}</td>
          </tr>
          <tr>
            <td>ob</td>
            <td>{JSON.stringify(obData)}</td>
          </tr>
          <tr>
            <td>agg_vol</td>
            <td>{JSON.stringify(aggVolData)}</td>
          </tr>
          <tr>
            <td>global</td>
            <td>{JSON.stringify(globalData)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default App;
