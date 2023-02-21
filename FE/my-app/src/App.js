import './App.css';
import React, { useEffect, useState } from 'react';

function App() {
  const [selectedTf, setSelectedTf] = useState("1h");
  const [whiteSoldiersData, setWhiteSoldiersData] = useState([]);
  const [russianDollData, setRussianDollData] = useState([]);
  const [bbTrendData, setBbTrendData] = useState([]);
  const [vwapData, setVwapData] = useState([]);
  const [atrData, setAtrData] = useState([]);
  const [aggVolData, setAggVolData] = useState([]);
  const [globalData, setGlobalData] = useState([]);
  const [tableData, setTableData] = useState([]);

  useEffect(() => {
    const requestOptions = {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Origin, X-Requested',
        'Access-Control-Allow-Methods': 'GET, POST, PATCH, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Credentials': 'true'
      }
    };

    fetch(`http://localhost:8000/white_soldiers?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setWhiteSoldiersData(data.data));

    fetch(`http://localhost:8000/russian_doll?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setRussianDollData(data.data));

    fetch(`http://localhost:8000/bb_trend?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setBbTrendData(data.data));

    fetch(`http://localhost:8000/vwap?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setVwapData(data.data));

    fetch(`http://localhost:8000/atr?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setAtrData(data.data));

    fetch(`http://localhost:8000/ob?symbol=BTCUSDT&tf=${selectedTf}`, requestOptions)
      .then(response => response.json())
      .then(data => setTableData(data.data));

    fetch('http://localhost:8000/agg_vol', requestOptions)
      .then(response => response.json())
      .then(data => setAggVolData(data.data));

    fetch('http://localhost:8000/global', requestOptions)
      .then(response => response.json())
      .then(data => setGlobalData(data.data));
  }, [selectedTf]);

  const handleTfChange = (event) => {
    setSelectedTf(event.target.value);
  }

  const tfOptions = ["1d", "6h", "4h", "2h", "1h", "30m", "15m", "5m", "1m"];

  return (
    <div className="App">
      <header className="App-header">
        <div>
          <select id="tf-select" value={selectedTf} onChange={handleTfChange}>
            {tfOptions.map((tf) => (
              <option key={tf} value={tf}>
                {tf}
              </option>
            ))}
          </select>
        </div>
        <div>
          whiteSoldiersData: {JSON.stringify(whiteSoldiersData[1] - 250)}
        </div>
        <div>
          russianDollData: {JSON.stringify(russianDollData[0] - russianDollData[1]/2 - russianDollData[2]/4 - russianDollData[3]/12 - russianDollData[4]/60)}
        </div>
        <div>
          Bollinger: {JSON.stringify(bbTrendData)}
        </div>
        <div>
          VWAP: {JSON.stringify(vwapData)}
        </div>
        <div>
          Avg True Range: {JSON.stringify(atrData)}
        </div>
        <div>
          {tableData.max && (
            <div>
              <label>Max Bids: {JSON.stringify(tableData.max.asks)}</label>
            </div>
          )}
        </div>
        <div>{aggVolData[0] && (
          <div>
            <label>24h price: {JSON.stringify(aggVolData[0].price_change_percentage_24h.toFixed(2))}</label>
          </div>
        )}
        </div>
        <div>{(globalData.data) && (
          <div>
            <label>24h market cap: {JSON.stringify(globalData.data.market_cap_change_percentage_24h_usd.toFixed(2))}</label>
            </div>
            )}        
        </div>
      </header>
    </div>
  );
}

export default App;