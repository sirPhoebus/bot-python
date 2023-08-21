import './App.css';
import React, { useEffect, useState, useMemo } from 'react';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTf, setSelectedTf] = useState("1h");
  const [selectedSymbol, setSelectedSymbol] = useState("BTCUSDT");
  const [whiteSoldiersData, setWhiteSoldiersData] = useState([]);
  const [russianDollData, setRussianDollData] = useState([]);
  const [bbTrendData, setBbTrendData] = useState([]);
  const [vwapData, setVwapData] = useState([]);
  const [atrData, setAtrData] = useState([]);
  const [aggVolData, setAggVolData] = useState([]);
  const [globalData, setGlobalData] = useState([]);
  const [tableData, setTableData] = useState([]);
  const url = "http://localhost:8000";
  const tfOptions = ["1d", "6h", "4h", "2h", "1h", "30m", "15m", "5m", "1m"];
  const symbolOptions = ["ETHUSDT", "BTCUSDT", "BNBUSDT"];

  const getPromises = (tf, symbol) => {
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

    return [
      fetch(url + `/white_soldiers?symbol=${symbol}&tf=${tf}`, requestOptions),
      fetch(url + `/russian_doll?symbol=${symbol}&tf=${tf}`, requestOptions),
      fetch(url + `/bb_trend?symbol=${symbol}&tf=${tf}`, requestOptions),
      fetch(url + `/vwap?symbol=${symbol}&tf=${tf}`, requestOptions),
      fetch(url + `/atr?symbol=${symbol}&tf=${tf}`, requestOptions),
      fetch(url + `/ob?symbol=${symbol}&tf=${tf}`, requestOptions),
      fetch(url + `/agg_vol`, requestOptions),
      fetch(url + `/global`, requestOptions)
    ];
  };

  const promises = useMemo(() => getPromises(selectedTf, selectedSymbol), [selectedTf, selectedSymbol]);

  const handleDropdownChange = (event) => {
    setIsLoading(true);
    if (event.target.id === 'tf-select') {
      setSelectedTf(event.target.value);
    } else {
      setSelectedSymbol(event.target.value);
    }
  };

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
        setIsLoading(false);
      });
  }, [promises, selectedTf, selectedSymbol]);

return (
    <div className="App">
      <header className="App-header">
        {isLoading ? (
          <div>Loading data... Please wait.</div>
        ) : (
          <>
            <div>
              <div>
                <font size="2">Pick your timeframe: </font>
                <select id="tf-select" value={selectedTf} onChange={handleDropdownChange}>
                  {tfOptions.map((tf) => (
                    <option key={tf} value={tf}>{tf}</option>
                  ))}
                </select>
              </div>
              <div>
                <font size="2">Pick your symbol: </font>
                <select id="symbol-select" value={selectedSymbol} onChange={handleDropdownChange}>
                  {symbolOptions.map((symbol) => (
                    <option key={symbol} value={symbol}>{symbol}</option>
                  ))}
                </select>
              </div>
              <font size="2"> for 500 candles</font>
            </div>
<table style={{ border: '1px solid #ddd', width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
  <tbody>
    <tr style={{ borderBottom: '1px solid #ddd' }}>
      <td style={{ padding: '8px' }}>White Soldiers:</td>
      <td style={{ padding: '8px', color: whiteSoldiersData[1] - 250 < 0 ? 'red' : 'green' }}>
        {whiteSoldiersData[1] - 250 ? whiteSoldiersData[1] - 250 : 'Loading...'}
      </td>
    </tr>
    <tr style={{ borderBottom: '1px solid #ddd' }}>
      <td style={{ padding: '8px' }}>(1h -> 1m):</td>
      <td style={{ padding: '8px' }}>{russianDollData[0]} , {russianDollData[1]} , {russianDollData[2]} , {russianDollData[3]} , {russianDollData[4]}</td>
    </tr>
    <tr style={{ borderBottom: '1px solid #ddd' }}>
      <td style={{ padding: '8px' }}>Bollinger:</td>
      <td style={{ padding: '8px' }}>{bbTrendData}</td>
    </tr>
    <tr style={{ borderBottom: '1px solid #ddd' }}>
      <td style={{ padding: '8px' }}>VWAP(MA5):</td>
      <td style={{ padding: '8px', color: vwapData === 'Above' ? 'green' : 'red' }}>{vwapData ? vwapData : 'Loading...'}</td>
    </tr>
    <tr style={{ borderBottom: '1px solid #ddd' }}>
      <td style={{ padding: '8px' }}>Avg True Range:</td>
      <td style={{ padding: '8px', color: atrData === 'downtrend' ? 'red' : 'green' }}>{atrData ? atrData : 'Loading...'}</td>
    </tr>
    <tr style={{ borderBottom: '1px solid #ddd' }}>
      <td style={{ padding: '8px' }}>Max Bids:</td>
      <td style={{ padding: '8px' }}>{tableData.max && tableData.max.asks}</td>
    </tr>
    <tr style={{ borderBottom: '1px solid #ddd' }}>
      <td style={{ padding: '8px' }}>Max Asks:</td>
      <td style={{ padding: '8px' }}>{tableData.max && tableData.max.bids}</td>
    </tr>
    <tr style={{ borderBottom: '1px solid #ddd' }}>
      <td style={{ padding: '8px' }}>24h btc price:</td>
      <td style={{ padding: '8px' }}>{aggVolData[0] && (aggVolData[0].price_change_percentage_24h.toFixed(2))}%</td>
    </tr>
    <tr style={{ borderBottom: '1px solid #ddd' }}>
      <td style={{ padding: '8px' }}>24h market cap:</td>
      <td style={{ padding: '8px' }}>{globalData.data && globalData.data.market_cap_change_percentage_24h_usd.toFixed(2)}%</td>
    </tr>
  </tbody>
</table>
          </>
        )}
      </header>
    </div>
  );
}

export default App;


