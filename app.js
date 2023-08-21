const express = require('express');
const app = express();
const bodyParser = require('body-parser');

app.use(bodyParser.json());
var users = [];
// User Schema
const userSchema = {
  type: 'object',
  properties: {
    id: { type: 'integer' },
    name: { type: 'string' },
    email: { type: 'string' },
  },
};

// Create a new user
app.post('/users', (req, res) => {
  const user = {
    id: req.body.id,
    name: req.body.name,
    email: req.body.email,
  };
  // Save the user to a database
  users.push(user);
  console.log(users);
  res.status(201).send(user);
});

// Get a list of all users
app.get('/users', (req, res) => {
  // Read the users from the database
  // ...
  res.send(users);
});

// Update a user
app.put('/users/:id', (req, res) => {
  const user = {
    id: req.params.id,
    name: req.body.name,
    email: req.body.email,
  };
  // Update the user in the database

  res.send(user);
});

// Delete a user
app.delete('/users/:id', (req, res) => {
  // Delete the user from the database
  // ...
  res.send('User deleted');
});


app.listen(3000, () => {
  console.log('Server started on port 3000');
});