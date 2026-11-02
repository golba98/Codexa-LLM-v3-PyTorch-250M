const $ = (s) => document.getElementById(s),
    fmt = (x, d = 3) =>
        typeof x === "number"
            ? x.toLocaleString(undefined, { maximumFractionDigits: d })
            : "—";
let stream,
    plots = [],
    current,
    lastFresh;
const safe = (node, text) => (node.textContent = text);
function rowsOf(run) {
    return run.rows
        .filter(
            (r) =>
                r.event === "step_completed" ||
                (!r.event && r.loss !== undefined),
        )
        .map((r) => ({ ...r, ...(r.data || {}) }));
}
function draw(id, rows, keys, colors) {
    const el = $(id);
    el.replaceChildren();
    if (!rows.length) {
        el.textContent = "No measurements recorded.";
        return;
    }
    const xs = [...new Set(rows.map((r) => r.step))].sort((a, b) => a - b),
        maps = keys.map(
            (k) => new Map(rows.map((r) => [r.step, r[k] ?? null])),
        );
    const data = [
        xs,
        ...maps.map((m, i) =>
            xs.map((x) => {
                let v = m.get(x) ?? null;
                return id === "memory" && v !== null ? v / 1073741824 : v;
            }),
        ),
    ];
    plots.push(
        new uPlot(
            {
                width: Math.max(180, el.clientWidth - 5),
                height: id === "loss" ? 265 : 180,
                scales: { x: { time: false } },
                series: [
                    {},
                    ...keys.map((k, i) => ({
                        label: k.replaceAll("_", " "),
                        stroke: colors[i],
                        width: 1.5,
                        spanGaps: false,
                        points: { show: k === "validation_loss", size: 5 },
                    })),
                ],
                axes: [
                    { stroke: "#778294", grid: { stroke: "#252b36" } },
                    {
                        stroke: "#778294",
                        grid: { stroke: "#252b36" },
                        size: 55,
                    },
                ],
                cursor: { drag: { x: true, y: false } },
            },
            data,
            el,
        ),
    );
}
function render(run) {
    const fresh = Date.now() - Date.parse(run.last?.timestamp || 0) < 15000;
    if (
        lastFresh === fresh &&
        current &&
        current.id === run.id &&
        current.records === run.records &&
        current.last?.session_id === run.last?.session_id &&
        current.last?.seq === run.last?.seq
    )
        return;
    lastFresh = fresh;
    current = run;
    plots.forEach((p) => p.destroy());
    plots = [];
    const rows = rowsOf(run),
        last = rows.at(-1) || {},
        meta = run.metadata.metadata || {},
        latest = run.metadata.latest || run.last || {},
        status = ["completed", "failed", "paused", "interrupted"].includes(
            latest.event,
        )
            ? latest.event
            : run.legacy
              ? "historical"
              : Date.now() - Date.parse(latest.timestamp || 0) > 15000
                ? "stale"
                : "running";
    safe($("name"), run.name);
    safe($("status"), status.toUpperCase());
    $("status").dataset.state = status;
    safe(
        $("subtitle"),
        `${meta.architecture || "Legacy metadata unavailable"} · ${fmt(meta.parameters, 0)} parameters · ${run.records.toLocaleString()} records${run.indexing ? " · indexing history…" : ""}${run.corrupt ? " · " + run.corrupt + " corrupt records" : ""}`,
    );
    safe(
        $("connection"),
        run.legacy
            ? "Archived legacy metrics"
            : Date.now() - Date.parse(latest.timestamp || 0) < 15000
              ? "Following recent telemetry"
              : "Saved history / no recent event",
    );
    $("cards").replaceChildren();
    for (const [label, value] of [
        ["Optimizer step", fmt(last.step, 0)],
        ["Tokens processed", fmt(last.tokens, 0)],
        ["Training loss", fmt(last.loss)],
        [
            "Validation loss",
            fmt(
                [...run.rows]
                    .reverse()
                    .find((r) => r.event === "validation_completed")?.data
                    ?.validation_loss ?? last.validation_loss,
            ),
        ],
        ["Tokens / second", fmt(last.tokens_per_second, 0)],
        ["Learning rate", last.lr?.toExponential(2) || "—"],
        ["Context / target", `${meta.model?.context || "—"} / 32,768`],
        [
            "Sequence length",
            fmt(last.sequence_length || meta.training?.sequence, 0),
        ],
    ]) {
        const div = document.createElement("div");
        div.className = "card";
        const a = document.createElement("span"),
            b = document.createElement("b");
        a.textContent = label;
        b.textContent = value;
        div.append(a, b);
        $("cards").append(div);
    }
    const validation = run.rows
        .filter((r) => r.event === "validation_completed")
        .map((r) => ({
            step: r.step,
            validation_loss: r.data.validation_loss,
        }));
    const merged = new Map(rows.map((r) => [r.step, { ...r }]));
    for (const v of validation)
        merged.set(v.step, { ...merged.get(v.step), ...v });
    draw(
        "loss",
        [...merged.values()],
        ["loss", "validation_loss"],
        ["#7aa2ff", "#4cc38a"],
    );
    draw("speed", rows, ["tokens_per_second"], ["#7aa2ff"]);
    draw("lr", rows, ["lr"], ["#e6b450"]);
    draw(
        "memory",
        rows,
        ["cuda_allocated_bytes", "cuda_reserved_bytes"],
        ["#7aa2ff", "#ba9cff"],
    );
    draw("gradient", rows, ["gradient_norm"], ["#4cc38a"]);
    $("gpu").replaceChildren();
    for (const [k, v] of [
        ["Device", meta.device || "Unavailable"],
        ["Precision", meta.precision || "Unavailable"],
        [
            "Attention",
            meta.model?.attention_backend || "Auto / kernel unrecorded",
        ],
        [
            "Allocated",
            last.cuda_allocated_bytes == null
                ? "Unavailable"
                : fmt(last.cuda_allocated_bytes / 1073741824) + " GiB",
        ],
        [
            "Reserved",
            last.cuda_reserved_bytes == null
                ? "Unavailable"
                : fmt(last.cuda_reserved_bytes / 1073741824) + " GiB",
        ],
        [
            "Utilization / temperature / power",
            last.gpu_system
                ? `${fmt(last.gpu_system.utilization_percent)}% / ${fmt(last.gpu_system.temperature_celsius)}°C / ${fmt(last.gpu_system.power_watts)} W (${last.gpu_system.scope})`
                : "Unavailable",
        ],
    ])
        addRow($("gpu"), k, v);
    $("contexts").replaceChildren();
    for (const n of [2048, 4096, 8192, 16384, 32768])
        addRow($("contexts"), n.toLocaleString(), contextLabel(n, meta));
    $("math").replaceChildren();
    for (const label of [
        "Grades 1–3",
        "Grades 4–6",
        "Grades 7–9",
        "Grades 10–12",
    ])
        addRow($("math"), label, "Inactive");
    $("checkpoints").replaceChildren();
    for (const r of run.rows
        .filter((r) => r.event === "checkpoint_saved")
        .slice(-5))
        addRow(
            $("checkpoints"),
            "Step " + r.step,
            fmt(r.data.size_bytes / 1048576) +
                " MiB · " +
                fmt(r.data.duration_seconds) +
                "s",
        );
    if (!$("checkpoints").children.length)
        $("checkpoints").textContent = "No checkpoint events recorded.";
    $("events").replaceChildren();
    for (const r of run.rows
        .filter((r) => r.event && r.event !== "step_completed")
        .slice(-20)
        .reverse()) {
        const div = document.createElement("div");
        div.className = "event";
        for (const t of [
            new Date(r.timestamp).toLocaleTimeString(),
            r.event.replaceAll("_", " "),
            "step " + r.step,
        ]) {
            const a = document.createElement("span");
            a.textContent = t;
            div.append(a);
        }
        $("events").append(div);
    }
}
function addRow(el, k, v) {
    const div = document.createElement("div");
    div.className = "row";
    const a = document.createElement("span"),
        b = document.createElement("div");
    a.textContent = k;
    b.textContent = v;
    div.append(a, b);
    el.append(div);
}
function follow(id) {
    document.getElementById("comparison")?.remove();
    $("compare").value = "";
    if (stream) stream.close();
    if (!id) return;
    history.replaceState(null, "", "?run=" + id);
    stream = new EventSource("/api/events?id=" + encodeURIComponent(id));
    stream.onopen = () =>
        safe($("connection"), "Connected · local telemetry/history");
    stream.onmessage = (e) => render(JSON.parse(e.data));
    stream.onerror = () =>
        safe($("connection"), "Disconnected · reconnecting automatically");
}
fetch("/api/runs")
    .then((r) => r.json())
    .then((runs) => {
        for (const r of runs) {
            for (const id of ["runs", "compare"]) {
                const o = document.createElement("option");
                o.value = r.id;
                o.textContent = r.name;
                $(id).append(o);
            }
        }
        if (runs.length) {
            $("runs").value =
                new URLSearchParams(location.search).get("run") || runs[0].id;
            follow($("runs").value);
        } else
            safe($("connection"), "No runs found. Start a Codexa smoke run.");
    })
    .catch(() => safe($("connection"), "Could not load runs"));
$("runs").onchange = (e) => follow(e.target.value);
$("compare").onchange = async (e) => {
    document.getElementById("comparison")?.remove();
    if (!e.target.value || !current) return;
    const other = await fetch("/api/run?id=" + e.target.value).then((r) =>
        r.json(),
    );
    const a = rowsOf(current).at(-1) || {},
        b = rowsOf(other).at(-1) || {};
    const panel = document.createElement("section");
    panel.className = "panel";
    panel.id = "comparison";
    document.getElementById("comparison")?.remove();
    const h = document.createElement("h2");
    h.textContent = "Experiment comparison · " + other.name;
    panel.append(h);
    addRow(
        panel,
        "Caution",
        "Dataset, hardware and sequence may differ; no capability delta inferred.",
    );
    for (const k of [
        "step",
        "tokens",
        "loss",
        "validation_loss",
        "tokens_per_second",
        "sequence_length",
    ])
        addRow(panel, k, fmt(a[k]) + " / " + fmt(b[k]));
    addRow(
        panel,
        "Configurations",
        JSON.stringify([
            current.metadata.metadata || {},
            other.metadata.metadata || {},
        ]),
    );
    $("events").parentElement.before(panel);
};
let resize;
window.addEventListener("resize", () => {
    clearTimeout(resize);
    resize = setTimeout(() => {
        if (current) {
            const saved = current;
            current = null;
            render(saved);
        }
    }, 200);
});

fetch("/api/context")
    .then((r) => r.json())
    .then((evidence) => {
        window.contextEvidence = evidence;
        if (current) {
            const saved = current;
            current = null;
            render(saved);
        }
    })
    .catch(() => {});

function contextLabel(n, meta) {
    const evidence = (window.contextEvidence || []).filter(
            (e) => e.length === n,
        ),
        inference = evidence.some(
            (e) => e.mode === "inference" && e.status === "ok",
        ),
        training = evidence.filter((e) => e.mode === "training");
    return inference
        ? training.length
            ? training.some((e) => e.status === "ok")
                ? "Inference + bounded updates measured"
                : "Inference measured; training OOM"
            : "Inference execution measured"
        : meta.model?.context === n
          ? "Configured"
          : "No execution evidence";
}
