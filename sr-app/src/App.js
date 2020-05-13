import React, { useState } from "react";
import logo from "./logo.svg";
import "./App.css";
import axios, { get, post } from "axios";

function App() {
  const [getFile, setFile] = useState();
  const handleSubmit = event => {
    event.preventDefault();

    const url = "http://127.0.0.1:5000/";
    const formData = new FormData();
    formData.append("file", getFile);
    formData.append("text", "getFile");
    const config = {
      headers: {
        "content-type": "multipart/form-data"
      }
    };
    return post(url, formData, config).then(() => {
      get(url);
    });
  };

  const handleChange = event => {
    setFile(event.target.files[0]);
  };

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
        <form onSubmit={handleSubmit}>
          <label htmlFor="myfile">Select a file:</label>
          <input
            type="file"
            id="myfile"
            name="myfile"
            onChange={handleChange}
          />
          <input type="submit" />
        </form>
        <img src="http://127.0.0.1:5000/return-files/" />
      </header>
    </div>
  );
}

export default App;
