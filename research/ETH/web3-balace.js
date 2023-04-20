const Web3 = require('web3')
const rpcURL = 'https://mainnet.infura.io/v3/7d99ce4493194f9fb0d92df48f61746c' // Your RPC URL goes here
const web3 = new Web3(rpcURL)
const address = '0x1483480eC53F948746b061b1AE9e1e47426A58DD' // Your account address goes here
web3.eth.getBalance(address, (err, wei) => {
  balance = web3.utils.fromWei(wei, 'ether')
})