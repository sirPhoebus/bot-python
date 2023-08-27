// src/StrategyBacktester.js

import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";
import { createChart } from 'lightweight-charts';



const StrategyBacktester = () => {
  const [symbol, setSymbol] = useState('');
  const [timeframe, setTimeframe] = useState('1h');
  const [strategy, setStrategy] = useState('bb_trend');
  const [startDate, setStartDate] = useState(new Date());
  const [endDate, setEndDate] = useState(new Date());
  const [data, setData] = useState([]);
  const chartContainerRef = useRef(null);
  
  const fetchData = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/${strategy}`, {
        params: { symbol, tf: timeframe }
      });
      setData(response.data.data.data);  // Note the extra .data here
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  useEffect(() => {
    if (chartContainerRef.current && data.length > 0) {
      const chart = createChart(chartContainerRef.current, { width: 700, height: 300 });
      const lineSeries = chart.addLineSeries();

      // Convert data to the format expected by TradingView
      const formattedData = data.map(item => ({
        time: item.open_time / 1000, // Convert to seconds if it's in milliseconds
        value: item.close_price
      }));

      lineSeries.setData(formattedData);

      // Enable auto-resize
      chart.applyOptions({
        layout: {
          backgroundColor: '#FFFFFF',
          textColor: 'rgba(33, 56, 77, 1)',
        },
        grid: {
          vertLines: {
            color: 'rgba(197, 203, 206, 0.5)',
          },
          horzLines: {
            color: 'rgba(197, 203, 206, 0.5)',
          },
        },
        crosshair: {
          mode: 1,
        },
        rightPriceScale: {
          borderColor: 'rgba(197, 203, 206, 0.8)',
        },
        timeScale: {
          borderColor: 'rgba(197, 203, 206, 0.8)',
        },
      });

      // Handle auto-resize
      function handleResize() {
        chart.resize(
          chartContainerRef.current.clientWidth,
          chartContainerRef.current.clientHeight
        );
      }

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
      };
    }
  }, [data]);

  return (
    <div>
      <h1>Strategy Backtester</h1>
      <div>
        <label>Symbol: </label>
        <input value={symbol} onChange={(e) => setSymbol(e.target.value)} />
      </div>
      <div>
        <label>Timeframe: </label>
        <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
          <option value="1h">1h</option>
          <option value="30m">30m</option>
          <option value="15m">15m</option>
          <option value="5m">5m</option>
          <option value="1m">1m</option>
        </select>
      </div>
      <div>
        <label>Strategy: </label>
        <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
          <option value="bb_trend">Bollinger Bands Trend</option>
          <option value="vwap">VWAP</option>
          <option value="atr">ATR</option>
          <option value="white_soldiers">White Soldiers</option>
          <option value="russian_doll">Russian Doll</option>
        </select>
      </div>
      <div>
        <label>Start Date: </label>
        <DatePicker selected={startDate} onChange={(date) => setStartDate(date)} />
      </div>
      <div>
        <label>End Date: </label>
        <DatePicker selected={endDate} onChange={(date) => setEndDate(date)} />
      </div>
      <button onClick={fetchData}>Backtest</button>

      <div ref={chartContainerRef} style={{ width: '100%', height: '300px' }}></div>

    </div>

  );
};

export default StrategyBacktester;
