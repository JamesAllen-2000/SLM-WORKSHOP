# SLM & Edge AI Workshop — Setup Instructions

**Please complete this BEFORE the workshop.** It takes about 45 minutes, most
of which is waiting for downloads.

If you arrive without this done, you will not be able to follow along — there
is no time to install during the session.

---

## Checklist

Tick these off as you go:

- [ ] **1.** Python 3.13 installed
- [ ] **2.** Git installed
- [ ] **3.** VS Code installed
- [ ] **4.** Windows long paths enabled
- [ ] **5.** Project downloaded
- [ ] **6.** Models downloaded and extracted
- [ ] **7.** Python packages installed
- [ ] **8.** `python verify_setup.py` shows all OK

---

## What your laptop needs

| | Minimum |
|---|---|
| Operating system | Windows 10 or 11 |
| RAM | 8 GB (16 GB is more comfortable) |
| Free disk space | **10 GB** |
| Internet | Needed for setup only — the workshop itself runs offline |

> No graphics card is required. Everything runs on your ordinary CPU. That is
> the whole point of the workshop.
>
> **Platform note for quantization:** Windows can use the bundled precompiled
> `llama.cpp` binaries. Linux and macOS require a native `llama-quantize`
> executable, either built from llama.cpp or supplied through the
> `LLAMA_QUANTIZE_PATH` environment variable.

---

# Step 1 — Install Python 3.13

**Download:** https://www.python.org/downloads/release/python-3130/

Scroll to the bottom and choose **"Windows installer (64-bit)"**.

### ⚠️ The one thing people get wrong

On the first screen of the installer, **tick "Add python.exe to PATH"** at the
bottom *before* clicking Install.

```
  ┌──────────────────────────────────────────────┐
  │  Install Python 3.13                         │
  │                                              │
  │   [ Install Now ]                            │
  │   [ Customize installation ]                 │
  │                                              │
  │   ☑ Use admin privileges when installing     │
  │   ☑ Add python.exe to PATH    <-- TICK THIS  │
  └──────────────────────────────────────────────┘
```

If you miss it, Windows will not find Python and nothing else will work. You
can fix it by running the installer again and choosing "Modify".

### Check it worked

Open **PowerShell** (press Start, type `powershell`, press Enter) and run:

```powershell
python --version
```

**Expected:** `Python 3.13.x`

> **Why version 3.13 specifically?** The workshop's packages are tested against
> it. Newer or older versions often have no ready-built package available and
> will try to compile from source, which fails. If you already have a different
> Python, install 3.13 alongside it — they can coexist.

---

# Step 2 — Install Git

**Download:** https://git-scm.com/download/win

Accept all the default options during installation.

### Check it worked

```powershell
git --version
```

**Expected:** `git version 2.x.x`

---

# Step 3 — Install VS Code

**Download:** https://code.visualstudio.com/

Accept the defaults. During installation, tick **"Add to PATH"** if offered.

After it opens, install the Python extension:

1. Click the **Extensions** icon in the left bar (four squares)
2. Search for **Python**
3. Install the one published by **Microsoft**

> VS Code is how you will read the code during the workshop. Every script is
> heavily commented and explains what it does — the session is much easier to
> follow with the files open beside the terminal.

---

# Step 4 — Enable long file paths

Windows refuses file paths longer than 260 characters. AI model folders are
deeply nested and will hit that limit.

Run this in PowerShell:

```powershell
git config --global core.longpaths true
```

**Also important:** put the project folder somewhere with a **short path**,
such as `C:\SLM-WORKSHOP`.

**Do not** put it inside OneDrive, or in a deep `Documents\...` folder. Those
paths are long before you even start, and downloads will fail.

---

# Step 5 — Get the project

```powershell
cd C:\
git clone https://github.com/JamesAllen-2000/SLM-WORKSHOP.git SLM-WORKSHOP
cd C:\SLM-WORKSHOP
```

This downloads roughly 250 MB and takes a few minutes.

> **No Git?** You can instead download the project as a ZIP from the GitHub
> page (green **Code** button → **Download ZIP**) and extract it to
> `C:\SLM-WORKSHOP`.

---

# Step 6 — Get the AI models

The models are too large for GitHub, so they are shared separately.

**Download `models.zip`** (about 2.4 GB) from the link your instructor sent.

### ⚠️ Extract it carefully — this is where most people go wrong

The zip already contains a folder called `models`. If you right-click and use
**Extract All**, Windows suggests a destination folder *also* called `models` —
and you end up with `models\models\`, which does not work.

**This is what you want:**

```
C:\SLM-WORKSHOP\
├── config.py
├── requirements.txt
├── models\              <-- the extracted folder
│   ├── it\
│   ├── embedder\
│   ├── adapter_20260905_143045\
│   └── gguf\
└── tests\
```

**This is the mistake to avoid:**

```
C:\SLM-WORKSHOP\
└── models\
    └── models\          <-- WRONG: one folder too deep
        ├── it\
        └── ...
```

### The safe way — copy and paste this

```powershell
cd C:\SLM-WORKSHOP
Expand-Archive -Path "$env:USERPROFILE\Downloads\models.zip" -DestinationPath . -Force
```

The `-DestinationPath .` means *"into the current folder"*. Because the zip
already carries its own `models` folder, this lands it in exactly the right
place. Adjust the path if your download went somewhere other than `Downloads`.

### Check it worked

```powershell
dir C:\SLM-WORKSHOP\models
```

**Expected:** four items — `it`, `embedder`, `adapter_...`, `gguf`

If instead you see a single item called `models`, you have the double-folder
problem. Fix it with:

```powershell
cd C:\SLM-WORKSHOP
Move-Item models\models\* models\ -Force
Remove-Item models\models
```

> **Google Drive will warn** *"Google Drive can't scan this file for viruses"*
> because the file is large. This is normal. Click **Download anyway**.
>
> **Download this a day or two early.** If forty people download the same file
> at once, Google Drive may temporarily block it.

Once extracted successfully, you can delete `models.zip` to save 2.4 GB.

---

# Step 7 — Install the Python packages

From inside the project folder:

```powershell
cd C:\SLM-WORKSHOP
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

This downloads about 2.5 GB and takes 5–20 minutes. Leave it running.

### What you should see

After `activate`, your prompt gains a `(venv)` prefix:

```
(venv) PS C:\SLM-WORKSHOP>
```

That means you are using the workshop's isolated set of packages. **You need to
run `.\venv\Scripts\activate` every time you open a new terminal.**

> ### If PowerShell blocks the activate script
> You may see *"running scripts is disabled on this system"*. Fix it with:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Answer `Y`, then try `.\venv\Scripts\activate` again.

> ### One package needs a special source
> `requirements.txt` begins with a line starting `--extra-index-url`. **Do not
> remove it.** One package (`llama-cpp-python`) is not published in the normal
> place, and without that line pip tries to compile it from C++ source — which
> fails unless you have Visual Studio installed.
>
> If you see **"Building wheel for llama-cpp-python"** scrolling past, stop and
> contact your instructor. It should download, not build.

---

# Step 8 — Verify everything

This is the important one:

```powershell
python verify_setup.py
```

### What success looks like

```
1. ENVIRONMENT
  [OK     ] Python version              3.13.x
  [OK     ] RAM                         16.0 GB

2. PYTHON PACKAGES
  [OK     ] PyTorch                     2.14.0+cpu
  [OK     ] Transformers                5.16.1
  [OK     ] llama-cpp-python            0.3.35
  ...

3. MODELS
  [OK     ] Base model                  models/it (1945 MB)
  [OK     ] Embedding model             models/embedder (88 MB)
  [OK     ] Fine-tuned adapter          models/adapter_...
  [OK     ] Quantized model (GGUF)      ...q4_k_m.gguf (777 MB)

RESULT: READY FOR THE WORKSHOP
```

**Every line must say `OK`, except one:** under *"4. RAG KNOWLEDGE BASE"* it is
normal to see `MISSING`. You build that in the first five minutes of the
workshop.

If anything else says `MISSING`, the tool prints the exact command that fixes
it. Run that command, then run `verify_setup.py` again.

---

# Common problems

### `python : The term 'python' is not recognized`
Python is not on your PATH. Re-run the Python installer, choose **Modify**, and
make sure **"Add python.exe to PATH"** is ticked. Then close and reopen
PowerShell.

### `ModuleNotFoundError: No module named 'torch'` (or trl, chromadb, ...)
Your virtual environment is not active. Look for `(venv)` at the start of your
prompt. If it is missing:
```powershell
cd C:\SLM-WORKSHOP
.\venv\Scripts\activate
```

### `ModuleNotFoundError: No module named 'llama_cpp'`
That package needs its special source:
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### `[WinError 206] The filename or extension is too long`
Your project folder is too deep. Move it to `C:\SLM-WORKSHOP`. If you cannot
move it, map a short drive letter:
```powershell
subst X: "C:\Your\Very\Long\Path\SLM WORKSHOP"
X:
```
(Undo later with `subst X: /D`.)

### `No quantized model found in models/gguf/`
Your `models` folder is incomplete or extracted to the wrong place. Check that
`C:\SLM-WORKSHOP\models\gguf\` exists and contains a `.gguf` file.

### Running scripts is disabled on this system
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

# Summary — all commands in one place

```powershell
# after installing Python 3.13, Git and VS Code

git config --global core.longpaths true

cd C:\
git clone https://github.com/JamesAllen-2000/SLM-WORKSHOP.git SLM-WORKSHOP
cd C:\SLM-WORKSHOP

# extract models.zip into this folder first, then:

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python verify_setup.py
```

---

# Before you close your laptop

Confirm you can see **`RESULT: READY FOR THE WORKSHOP`**.

If you cannot get there, contact your instructor **before** the day — with the
error message you are seeing. Most problems take two minutes to fix by email
and half an hour to fix in a full room.

See you at the workshop.
