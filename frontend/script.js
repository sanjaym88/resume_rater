document.addEventListener("DOMContentLoaded", function () {
  const responseBox = document.getElementById("response");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const chatOutput = document.getElementById("chatOutput");
  const scoreDisplay = document.getElementById("scoreDisplay");

  document.getElementById("uploadForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const formData = new FormData();
    formData.append("resume", document.getElementById("resume").files[0]);
    formData.append("jd", document.getElementById("jd").files[0]);
    formData.append("prompt", document.getElementById("prompt").files[0]);
    formData.append("session_id", "default");

    responseBox.textContent = "Rating in progress...";
    chatOutput.innerHTML = "";

    try {
      const res = await fetch("/evaluate", {
        method: "POST",
        body: formData,
      });

      const result = await res.json();

      scoreDisplay.textContent = "Score: " + result.score;
      const cleanedReason = result.reason.replace(/Score:\s*\d+/gi, "").trim();
      responseBox.textContent = cleanedReason;
      addMessage(cleanedReason, false);
      chatForm.style.display = "flex";
    } catch (err) {
      responseBox.textContent = "Error: " + err.message;
    }
  });

  chatForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    const msg = chatInput.value.trim();
    if (!msg) return;

    addMessage(msg, true);
    chatInput.value = "";

    const formData = new FormData();
    formData.append("user_input", msg);
    formData.append("session_id", "default");

    try {
      const res = await fetch("/chat", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      addMessage(data.response, false);
    } catch (err) {
      addMessage("Error: " + err.message, false);
    }
  });

  document.getElementById("batchForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const formData = new FormData();
    const resumeFiles = document.getElementById("batchResumes").files;
    for (let i = 0; i < resumeFiles.length; i++) {
      formData.append("resumes", resumeFiles[i]);
    }
    formData.append("jd", document.getElementById("batchJd").files[0]);
    formData.append("prompt", document.getElementById("batchPrompt").files[0]);

    const batchResults = document.getElementById("batchResults");
    batchResults.textContent = "Ranking resumes...";

    try {
      const res = await fetch("/batch-evaluate", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      renderBatchResults(data.results);
    } catch (err) {
      batchResults.textContent = "Error: " + err.message;
    }
  });

  function addMessage(text, isUser) {
    const chatOutput = document.getElementById("chatOutput");
    const parts = isUser ? [text] : text.split(/\n+|\* /);

    parts.forEach((part) => {
      const clean = part.trim();
      if (!clean) return;

      const messageDiv = document.createElement("div");
      messageDiv.className = "message " + (isUser ? "user-message" : "ai-message");
      messageDiv.textContent = isUser ? clean : (clean.startsWith("•") ? clean : "• " + clean);

      chatOutput.appendChild(messageDiv);
    });

    chatOutput.scrollTop = chatOutput.scrollHeight;
  }
});

function renderBatchResults(results) {
  const container = document.getElementById("batchResults");
  container.innerHTML = "";

  const table = document.createElement("table");
  table.style.width = "100%";
  table.style.borderCollapse = "collapse";
  table.innerHTML = `
    <tr style="text-align:left; border-bottom: 1px solid #444;">
      <th style="padding: 0.5rem;">Candidate</th>
      <th style="padding: 0.5rem;">Score</th>
      <th style="padding: 0.5rem;">Summary</th>
      <th style="padding: 0.5rem;">Action</th>
    </tr>
  `;

  results.forEach((r) => {
    const row = document.createElement("tr");
    row.style.borderBottom = "1px solid #333";
    row.innerHTML = `
      <td style="padding: 0.5rem;">${r.filename}</td>
      <td style="padding: 0.5rem;">${r.score}</td>
      <td style="padding: 0.5rem;">${r.reason}</td>
      <td style="padding: 0.5rem;"><button class="chat-candidate-btn">Chat</button></td>
    `;
    table.appendChild(row);

    const chatBtn = row.querySelector(".chat-candidate-btn");
    chatBtn.addEventListener("click", () => openBatchChat(r.filename));
  });

  container.appendChild(table);

  const exportBtn = document.createElement("button");
  exportBtn.textContent = "Export CSV";
  exportBtn.style.marginTop = "1rem";
  exportBtn.addEventListener("click", () => exportCSV(results));
  container.appendChild(exportBtn);
}

function exportCSV(results) {
  let csv = "Candidate,Score,Summary\n";
  results.forEach((r) => {
    const safeSummary = r.reason.replace(/"/g, '""').replace(/\n/g, " ");
    csv += `"${r.filename}",${r.score},"${safeSummary}"\n`;
  });

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "resume_rankings.csv";
  a.click();
  URL.revokeObjectURL(url);
}

let currentBatchSessionId = null;

function openBatchChat(filename) {
  currentBatchSessionId = "batch_" + filename;
  document.getElementById("batchChatSection").style.display = "block";
  document.getElementById("batchChatLabel").textContent = "Chatting about: " + filename;
  document.getElementById("batchChatOutput").innerHTML = "";
  document.getElementById("batchChatSection").scrollIntoView({ behavior: "smooth" });
}

document.getElementById("batchChatForm").addEventListener("submit", async function (e) {
  e.preventDefault();
  const input = document.getElementById("batchChatInput");
  const msg = input.value.trim();
  if (!msg || !currentBatchSessionId) return;

  const output = document.getElementById("batchChatOutput");

  const userMsg = document.createElement("div");
  userMsg.className = "message user-message";
  userMsg.textContent = msg;
  output.appendChild(userMsg);
  input.value = "";

  const formData = new FormData();
  formData.append("user_input", msg);
  formData.append("session_id", currentBatchSessionId);

  try {
    const res = await fetch("/chat", { method: "POST", body: formData });
    const data = await res.json();
    const aiMsg = document.createElement("div");
    aiMsg.className = "message ai-message";
    aiMsg.textContent = data.response;
    output.appendChild(aiMsg);
  } catch (err) {
    const errMsg = document.createElement("div");
    errMsg.className = "message ai-message";
    errMsg.textContent = "Error: " + err.message;
    output.appendChild(errMsg);
  }

  output.scrollTop = output.scrollHeight;
});

document.addEventListener("mousemove", function (e) {
  const light = document.getElementById("spotlight");
  if (light) {
    light.style.left = e.clientX + "px";
    light.style.top = e.clientY + "px";
  }
});