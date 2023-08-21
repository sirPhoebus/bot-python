// Setup: npm install alchemy-sdk
const { Alchemy, Network } = require("alchemy-sdk");
const config = {
  apiKey: "ScnZCJCkhgnc-6HfDN7K0mGtvZTKhc8v",
  network: Network.ETH_MAINNET,
};
const alchemy = new Alchemy(config);
const numDecimals = 18;

const main = async () => {
  // Wallet address
  const address = "0xef0dcc839c1490cebc7209baa11f46cfe83805ab";

  // Get token balances
  const balances = await alchemy.core.getTokenBalances(address);

  // Remove tokens with zero balance
  const nonZeroBalances = balances.tokenBalances.filter((token) => {
    return token.tokenBalance !== 0;
  });

  console.log(`Token balances of ${address} \n`);

  // Counter for SNo of final output
  let i = 1;

  // Loop through all tokens with non-zero balance
  for (let token of nonZeroBalances) {
    // Get balance of token
    let balance = token.tokenBalance;

    // Get metadata of token
    const metadata = await alchemy.core.getTokenMetadata(token.contractAddress);

    // Compute token balance in human-readable format
    balance = balance / Math.pow(10, metadata.decimals);
    
    balance = balance.toFixed(2);
    // balance = (parseInt(balance) / 10 ** numDecimals).toFixed(2);
    balance = parseInt(balance) / 10**6;
    // Print name, balance, and symbol of token
    console.log(`${i++}. ${metadata.name}: ${balance} ${metadata.symbol}`);
  }
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
