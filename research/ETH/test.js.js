// var Web3 = require('web3');
// var provider = 'https://mainnet.infura.io/v3/7d99ce4493194f9fb0d92df48f61746c';
// var web3Provider = new Web3.providers.HttpProvider(provider);
// var web3 = new Web3(web3Provider);
// web3.eth.getBlockNumber().then((result) => {
//   console.log("Latest Ethereum Block is ",result);
// });

//ubuntu OK
// curl https://mainnet.infura.io/v3/7d99ce4493194f9fb0d92df48f61746c -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"eth_getBalance","params":["0x1483480eC53F948746b061b1AE9e1e47426A58DD", "latest"],"id":1}'

//quick node
// var myHeaders = new Headers();
// myHeaders.append("Content-Type", "application/json");

// var raw = JSON.stringify({
//   "method": "eth_getBalance",
//   "params": [
//     "0x00000000219ab540356cBB839Cbe05303d7705Fa",
//     "latest"
//   ],
//   "id": 1,
//   "jsonrpc": "2.0"
// });

// var requestOptions = {
//   method: 'POST',
//   headers: myHeaders,
//   body: raw,
//   redirect: 'follow'
// };

// fetch("https://mainnet.infura.io/v3/7d99ce4493194f9fb0d92df48f61746c", requestOptions)
//   .then(response => response.text())
//   .then(result => console.log(result))
//   .catch(error => console.log('error', error));

// Request


// const axios = require('axios/dist/node/axios.cjs');

// const apiKey = "6HfDN7K0mGtvZTKhc8v"
// const options = {
//     method: "POST",
//     url: `https://eth-mainnet.g.alchemy.com/v2/${apiKey}`,
//     headers: { accept: "application/json", "content-type": "application/json" },
//     data: {
//       id: 1,
//       jsonrpc: "2.0",
//       method: "eth_getBlockTransactionCountByNumber",
//       params: "finalized",
//     },
//   };
  
//   axios
//     .request(options)
//     .then(function (response) {
//          console.log(response.data);
//     })
//     .catch(function (error) {
//       console.error(error);
//     });


// async function doPostRequest() { 
// let payload = { name: 'John Doe', occupation: 'gardener' }; 
// let res = await axios.post('http://httpbin.org/post', payload); 
// let data = res.data; 
// console.log(data); 
// } 
// doPostRequest(); 
