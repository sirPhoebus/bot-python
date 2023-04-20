//SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract MyBalance {
    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
