// App.js
import React, { useState, useEffect } from 'react';

const App = () => {
  const [uuid, setUUID] = useState('');

  const handleUUIDChange = (newUUID) => {
    // Ensure that newUuid is a string
    if (typeof newUUID === 'string') {
      setUUID(newUUID);
    } else {
      console.error('received invalid uuid:', newUUID);
    }
  };

  const [inputText, setInputText] = useState('');

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleButtonClick = async () => {
    try {
      const response = await fetch('http://localhost:5600', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputText }),
      });

      const uuid = await response.json().then((data) => data.uuid);
      handleUUIDChange(uuid);
    } catch (error) {
      console.error('error fetching UUID:', error);
    }
  };

  const [sseData, setSSEData] = useState([]);

  useEffect(() => {

    setSSEData([]);
    if (uuid === '') {
      return;
    }
  
    const sse = new EventSource(`http://localhost:5600?uuid=${uuid}`);

    // Event listener for SSE messages
    sse.onmessage = event => {
      setSSEData((prev) => [...prev, event.data]);
    };

    sse.onerror = event => {
      sse.close();
    };

    return () => sse.close();
  }, [uuid]);

  return (
    <div>
      <h1>Customized Chatbot</h1>
      <h2>Question:</h2>
      <div>
        <input type="text" value={inputText} onChange={handleInputChange} />
        <button onClick={handleButtonClick}>submit</button>
      </div>
      <div>
        <h2>Answer:</h2>
        <p>{sseData}</p>
      </div>
    </div>
  );
};

export default App;
