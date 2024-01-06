// App.js
import React, { useState, useEffect } from 'react';

const App = () => {
  const [uuid, setUUID] = useState('');

  const handleUUIDChange = (newUUID) => {
    // Ensure that newUuid is a string
    if (typeof newUUID === 'string') {
      setUUID(newUUID);
      console.log('set uuid:', newUUID, 'Type', typeof newUUID)
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
        body: JSON.stringify({ text: inputText }),
      });

      const uuid = await response.json().then((data) => data.uuid);
      console.log('receive uuid:', uuid, 'Type', typeof uuid)
      handleUUIDChange(uuid);
    } catch (error) {
      console.error('Error fetching UUID:', error);
    }
  };

  const [sseData, setSSEData] = useState([]);

  useEffect(() => {

    if (uuid === '') {
      return;
    }
  
    const sse = new EventSource('http://localhost:5600?uuid=' + uuid);
    console.log('send uuid:', uuid, 'Type', typeof uuid)

    // Event listener for SSE messages
    sse.onmessage = (event) => {
      console.log('sse data:', event.data);
      setSSEData((prev) => [...prev, event.data]);
    };

    // Event listener for SSE opens
    sse.onopen = (event) => {
      console.log('sse opened:', event);
    };

    // Event listener for SSE errors
    sse.onerror = (error) => {
      console.error('sse failed:', error);
      sse.close();
    };

    return () => sse.close();
  }, [uuid]);

  return (
    <div>
      <h1>SSE App</h1>
      <div>
        <input type="text" value={inputText} onChange={handleInputChange} />
        <button onClick={handleButtonClick}>submit</button>
      </div>
      <div>
        <h2>SSE Data:</h2>
        <p>{sseData}</p>
      </div>
    </div>
  );
};

export default App;
