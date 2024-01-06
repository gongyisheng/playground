// App.js
import React, { useState, useEffect } from 'react';

const BASE_URL = 'http://localhost:5600';
const ENDPOINT_META = BASE_URL + '/meta';
const ENDPOINT_CHAT = BASE_URL + '/chat';

const App = () => {
  var [model, setModel] = useState('fetching...');
  const getMeta = async () => {
    try {
      const response = await fetch(ENDPOINT_META, {
        method: 'GET',
      });
      model = await response.json().then((data) => data.model);
      setModel(model);
    } catch (error) {
      console.error('error fetching meta:', error);
      setModel('unknown');
    }
  };
  getMeta();

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
      const response = await fetch(ENDPOINT_CHAT, {
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

  const [sseData, setSSEData] = useState('');

  useEffect(() => {

    setSSEData('');
    if (uuid === '') {
      return;
    }
  
    const sse = new EventSource(`${ENDPOINT_CHAT}?uuid=${uuid}`);

    // Event listener for SSE messages
    sse.onmessage = event => {
      setSSEData((prev) => prev + JSON.parse(event.data).content);
    };


    sse.onerror = event => {
      sse.close();
    };

    return () => sse.close();
  }, [uuid]);

  return (
    <div>
      <h1>Customized Chatbot [{model}]</h1>
      <h2>Question:</h2>
      <div>
        <input type="text" value={inputText} onChange={handleInputChange} />
        <button onClick={handleButtonClick}>submit</button>
      </div>
      <div>
        <h2>Answer:</h2>
        <div>{sseData}</div>
      </div>
    </div>
  );
};

export default App;
