const format = (value, digits = 2) => value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
fetch("latest.json").then((r) => r.json()).then((data) => {
  document.querySelector("#disclaimer").textContent = data.disclaimer;
  document.querySelector("#status").innerHTML = `<h2>${data.current_regime ?? "Awaiting first session"}</h2><p>Confidence: ${format(data.regime_confidence)} · Active expert: ${data.active_regime_model ?? "—"}</p>`;
  document.querySelector("#metrics").innerHTML = Object.entries(data.metrics).map(([name, metric]) => `<tr><td>${name.replaceAll("_", " ")}</td><td>${format(metric.cumulative_return)}</td><td>${metric.sharpe?.toFixed(2) ?? "—"}</td><td>${format(metric.max_drawdown)}</td></tr>`).join("");
  document.querySelector("#decisions").innerHTML = data.recent_decisions.map((d) => `<li>${d.date}: ${d.regime}; static ${format(d.static_target, 0)}, regime ${format(d.regime_target, 0)}</li>`).join("");
}).catch(() => { document.querySelector("#disclaimer").textContent = "Experiment feed is not available yet."; });
