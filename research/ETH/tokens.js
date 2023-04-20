// Setup: npm install alchemy-sdk
const { Alchemy, Network } = require("alchemy-sdk");

const config = {
    apiKey: "ScnZCJCkhgnc-6HfDN7K0mGtvZTKhc8v",
  network: Network.ETH_MAINNET,
};
const alchemy = new Alchemy(config);

const main = async () => {
  // Wallet address
  const address = "0xef0dcc839c1490cebc7209baa11f46cfe83805ab";

  // Get token balances
  const balances = await alchemy.core.getTokenBalances(address);

  console.log(`The balances of ${address} address are:`, balances);
};

const runMain = async () => {
  try {
    await main();
    process.exit(0);
  } catch (error) {
    console.log(error);
    process.exit(1);
  }
};

runMain();
