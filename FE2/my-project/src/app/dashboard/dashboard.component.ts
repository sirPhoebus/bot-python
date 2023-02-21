import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Component({
  selector: 'app-dashboard',
  templateUrl: `dashboard.component.html`
})
export class DashboardComponent implements OnInit {
  data: any[] = [];

  constructor(private http: HttpClient) { }

  ngOnInit() {
    const url = "http://localhost:8000";
    const tf = "1d";
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Origin, X-Requested',
      'Access-Control-Allow-Methods': 'GET, POST, PATCH, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Credentials': 'true'
    });

    this.getData(url + `/white_soldiers?symbol=BTCUSDT&tf=${tf}`, headers).subscribe(data => {
      this.data[0] = data;
    });
    this.getData(url + `/russian_doll?symbol=BTCUSDT&tf=${tf}`, headers).subscribe(data => {
      this.data[1] = data;
    });
    this.getData(url + `/bb_trend?symbol=BTCUSDT&tf=${tf}`, headers).subscribe(data => {
      this.data[2] = data;
    });
    this.getData(url + `/vwap?symbol=BTCUSDT&tf=${tf}`, headers).subscribe(data => {
      this.data[3] = data;
    });
    this.getData(url + `/atr?symbol=BTCUSDT&tf=${tf}`, headers).subscribe(data => {
      this.data[4] = data;
    });
    this.getData(url + `/ob?symbol=BTCUSDT&tf=${tf}`, headers).subscribe(data => {
      this.data[5] = data;
    });
    this.getData(url + '/agg_vol', headers).subscribe(data => {
      this.data[6] = data;
    });
    this.getData(url + '/global', headers).subscribe(data => {
      this.data[7] = data;
    });
  }

  private getData(url: string, headers: HttpHeaders) {
    return this.http.get<any[]>(url, { headers });
  }
}
