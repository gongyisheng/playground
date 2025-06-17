// DOM elements
const responseList = document.getElementById('responseList');
const downloadBtn = document.getElementById('downloadBtn');
const clearBtn = document.getElementById('clearBtn');

// Load responses when popup opens
document.addEventListener('DOMContentLoaded', () => {
  loadResponses();
});

// Listen for new responses
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'newResponse') {
    addResponseToList(message.data);
  }
});

// Load responses from storage
function loadResponses() {
  chrome.runtime.sendMessage({ type: 'getResponses' }, (response) => {
    if (response.responses && response.responses.length > 0) {
      responseList.innerHTML = '';
      response.responses.forEach(addResponseToList);
    }
  });
}

// Add a response to the list
function addResponseToList(response) {
  const responseItem = document.createElement('div');
  responseItem.className = 'response-item';
  
  const url = document.createElement('div');
  url.className = 'response-url';
  url.textContent = response.url;
  
  const meta = document.createElement('div');
  meta.className = 'response-meta';
  meta.textContent = `${response.method} | ${response.statusCode} | ${new Date(response.timestamp).toLocaleString()}`;
  
  responseItem.appendChild(url);
  responseItem.appendChild(meta);
  
  // Add click handler to show response body
  responseItem.addEventListener('click', () => {
    const responseBody = document.createElement('pre');
    responseBody.style.marginTop = '10px';
    responseBody.style.padding = '10px';
    responseBody.style.backgroundColor = '#f5f5f5';
    responseBody.style.borderRadius = '4px';
    responseBody.style.overflow = 'auto';
    responseBody.style.maxHeight = '200px';
    
    try {
      // Try to parse and format JSON
      const jsonData = JSON.parse(response.responseBody);
      responseBody.textContent = JSON.stringify(jsonData, null, 2);
    } catch {
      // If not JSON, show as is
      responseBody.textContent = response.responseBody;
    }
    
    // Toggle response body
    if (responseItem.contains(responseBody)) {
      responseItem.removeChild(responseBody);
    } else {
      responseItem.appendChild(responseBody);
    }
  });
  
  responseList.appendChild(responseItem);
}

// Download all responses
downloadBtn.addEventListener('click', () => {
  chrome.runtime.sendMessage({ type: 'downloadResponses' });
});

// Clear all responses
clearBtn.addEventListener('click', () => {
  if (confirm('Are you sure you want to clear all captured responses?')) {
    chrome.runtime.sendMessage({ type: 'clearResponses' }, () => {
      responseList.innerHTML = '<div class="no-responses">No responses captured yet</div>';
    });
  }
}); 