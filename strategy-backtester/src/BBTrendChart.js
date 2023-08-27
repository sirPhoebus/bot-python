// src/BBTrendChart.js

import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

const BBTrendChart = ({ data }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();  // Clear previous chart

    const width = 1900;
    const height = 1200;
    const margin = { top: 20, right: 20, bottom: 30, left: 50 };

    const x = d3.scaleTime()
      .domain(d3.extent(data, d => new Date(d.open_time)))
      .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
      .domain([d3.min(data, d => d.lower_bound), d3.max(data, d => d.upper_bound)])
      .nice()
      .range([height - margin.bottom, margin.top]);

      const xAxis = g => g
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(width / 80).tickSizeOuter(0).tickFormat(d3.timeFormat("%H:%M")))
      .call(g => g.select(".domain").remove());

    const yAxis = g => g
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y))
      .call(g => g.select(".domain").remove());

    const zoom = d3.zoom()
      .scaleExtent([1, 32])
      .translateExtent([[margin.left, -Infinity], [width - margin.right, Infinity]])
      .on("zoom", zoomed);

    svg.append("clipPath")
      .attr("id", "clip")
      .append("rect")
      .attr("x", margin.left)
      .attr("y", margin.top)
      .attr("width", width - margin.left - margin.right)
      .attr("height", height - margin.top - margin.bottom);

    const line = d3.line()
      .defined(d => !isNaN(d.close_price))
      .x(d => x(new Date(d.open_time)))
      .y(d => y(d.close_price));

    svg.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "#8884d8")
      .attr("stroke-width", 1.5)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("clip-path", "url(#clip)")
      .attr("d", line);

    // Add more lines for upper_bound, lower_bound, and basis similarly...

    svg.append("g")
      .call(xAxis);

    svg.append("g")
      .call(yAxis);

    svg.call(zoom);

    function zoomed(event) {
      x.range([margin.left, width - margin.right].map(d => event.transform.applyX(d)));
      svg.selectAll("path").attr("d", line);
      svg.select(".x-axis").call(xAxis);
    }

  }, [data]);

  return (
    <svg ref={svgRef} viewBox={`0 0 1900 1200`} width="1900" height="1200"></svg>
  );
};

export default BBTrendChart;
