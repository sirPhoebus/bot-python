import './App.css';
import React, { useEffect, useState, useMemo } from 'react';

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
  const url = "http://localhost:8000"
  const tfOptions = ["1d", "6h", "4h", "2h", "1h", "30m", "15m", "5m", "1m"];
  
  
  const getPromises = (tf) => {
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
    
    return [      fetch(url +`/white_soldiers?symbol=BTCUSDT&tf=${tf}`, requestOptions),      
    fetch(url +`/russian_doll?symbol=BTCUSDT&tf=${tf}`, requestOptions),      
    fetch(url +`/bb_trend?symbol=BTCUSDT&tf=${tf}`, requestOptions),      
    fetch(url +`/vwap?symbol=BTCUSDT&tf=${tf}`, requestOptions),      
    fetch(url +`/atr?symbol=BTCUSDT&tf=${tf}`, requestOptions),      
    fetch(url +`/ob?symbol=BTCUSDT&tf=${tf}`, requestOptions),      
    fetch(url +`/agg_vol`, requestOptions),      
    fetch(url +`/global`, requestOptions)    ];
  };

  const promises = useMemo(() => getPromises(selectedTf), [selectedTf]);

  useEffect(() => {
    Promise.all(promises)
      .then(responses => Promise.all(responses.map(response => response.json())))
      .then(data => {
        setWhiteSoldiersData(data[0].data);
        setRussianDollData(data[1].data);
        setBbTrendData(data[2].data);
        setVwapData(data[3].data);
        setAtrData(data[4].data);
        setTableData(data[5].data);
        setAggVolData(data[6].data);
        setGlobalData(data[7].data);
      });
  }, [promises, selectedTf]);

  const handleTfChange = (event) => {
    const tf = event.target.value;
    setSelectedTf(tf);
  }
  

  return (
    <div className="App">
      <header className="App-header">
        <div><font size="2">Pick your timeframe: </font>
          <select id="tf-select" value={selectedTf} onChange={handleTfChange}>
            {tfOptions.map((tf) => (
              <option key={tf} value={tf}>
                {tf}
              </option>
            ))}
          </select><font size="2"> for 500 candles</font>
        </div>
        <div>
          White Soldiers: {whiteSoldiersData[1] - 250} <br />
          Russian Doll: {(russianDollData[0] - russianDollData[1] / 2 - russianDollData[2] / 4 - russianDollData[3] / 12 - russianDollData[4] / 60).toFixed(2)}<br />
          Bollinger: {bbTrendData}<br />
          VWAP(MA5): {vwapData}<br />
          Avg True Range: {atrData}<br />
          Max Bids: {tableData.max && tableData.max.asks}<br />
          Max Asks: {tableData.max && tableData.max.bids}<br />
          24h btc price: {aggVolData[0] && (aggVolData[0].price_change_percentage_24h.toFixed(2))}%<br />
          24h market cap: {globalData.data && globalData.data.market_cap_change_percentage_24h_usd.toFixed(2)}%
        </div>
      </header>
    </div>
  );
}

export default App;