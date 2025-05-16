const http = require("http");
const express = require("express");
require("dotenv").config();
const PORT = process.env.PORT || 3007;
// const bodyParser = require("body-parser");
const app = express();
const fs = require("fs");
const crypto = require("crypto");
const cors = require("cors");
const server = http.createServer(app);
const path = require("path");
const initializeDB = require("./InitialiseDb/index");
const multer = require("multer");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
// const { v4: uuidv4 } = require("uuid");
const { User } = require("./models/models");
const { execFile, spawn } = require("child_process");

app.use(cors());
app.use(express.json());
app.use("/uploads", express.static(path.join(__dirname, "uploads")));
initializeDB();

const uploadDir = path.join(__dirname, "uploads/faces/images", { recursive: true });
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

const clearTargetFolder = (req, res) => {
  let { rollno, name } = req.body;
  if (!rollno || !name) return res.status(400).send("Missing rollno or name");
  name = name.toLowerCase();
  const folderName = `${rollno}_${name}`.replace(/\s+/g, "_");
  const targetPath = path.join(__dirname, "uploads/faces/images", folderName);

  // Check and delete existing folder contents
  if (fs.existsSync(targetPath)) {
    fs.readdirSync(targetPath).forEach((file) => {
      const filePath = path.join(targetPath, file);
      if (fs.lstatSync(filePath).isFile()) {
        fs.unlinkSync(filePath); // delete file
      }
    });
  }

  req.uploadPath = targetPath; // pass to multer if needed
  next();
};

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const { rollno, name } = req.body;
    const folderName = `${rollno}_${name}`.replace(/\s+/g, "_"); // remove spaces if any
    const uploadPath = path.join(__dirname, "uploads/faces/images", folderName);
    console.log("jesvbjbdbdrbndjnb");
    fs.mkdirSync(uploadPath, { recursive: true }); // create folder if it doesn't exist

    cb(null, uploadPath);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    const ext = path.extname(file.originalname);
    cb(null, uniqueSuffix + ext);
  },
});

const upload = multer({ storage });

const reportVideoStorage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadPath = path.join(__dirname, "uploads/test-video");
    console.log(uploadPath);
    fs.mkdirSync(uploadPath, { recursive: true }); // create folder if it doesn't exist
    console.log("ejnejjv");
    cb(null, uploadPath);
  },
  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname);
    console.log("nbdbjn");
    cb(null, file.originalname);
  },
});

const uploadVideoStorage = multer({ storage: reportVideoStorage });

const attendenceImage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadPath = path.join(__dirname, "uploads/test-image");
    console.log(uploadPath);
    fs.mkdirSync(uploadPath, { recursive: true }); // create folder if it doesn't exist
    console.log("ejnejjv");
    cb(null, uploadPath);
  },
  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname);
    console.log("nbdbjn");
    cb(null, file.originalname);
  },
});

const uploadAttendenceImage = multer({ storage: attendenceImage });

app.post("/api/login", async (req, res) => {
  const { email, password } = req.body;
  console.log(email, password);
  try {
    let role;
    let person = await User.findOne({ email });
    console.log(person);

    if (person === null) {
      return res.status(400).send("User not found");
    }
    // const isMatched = password === person.password;

    const isMatched = await bcrypt.compare(password, person.password);
    // console.log("isMatched", isMatched);
    console.log(role);
    if (isMatched) {
      const payload = {
        email: email,
        role: person.role,
        userId: person.userId,
      };
      const token = jwt.sign(payload, "MY_SECRET_TOKEN");
      res.send({ token });
    } else {
      // res.status(401).send("incorrect password");
      return res.status(400).send("incorrect password");

      // res.redirect("/login");
    }
  } catch (e) {
    console.log(e);
    res.status(500).send(e);
  }
});

app.post(
  "/api/register-face",
  //   express.urlencoded({ extended: true }), // to parse form body
  upload.array("images[]"),
  async (req, res) => {
    let { name, rollno } = req.body;
    const files = req.files;
    name = name.toLowerCase();
    if (!name || !rollno || !files || files.length === 0) {
      return res.status(400).send("Missing fields or files.");
    }

    try {
      // Process all files in parallel
      const results = await Promise.all(
        files.map((file) => {
          return new Promise((resolve, reject) => {
            const args = [file.path, file.filename, rollno];

            execFile(
              "python",
              ["scripts/process_face.py", ...args],
              (error, stdout, stderr) => {
                if (error) {
                  return reject(stderr || error.message);
                }
                if (stdout.includes("MULTIPLE_FACES")) {
                  return reject(`Multiple faces found in ${file.originalname}`);
                }
                resolve(stdout.trim());
              }
            );
          });
        })
      );
      const inputDir = path.join(
        __dirname,
        `uploads/faces/images/${rollno}_${name}`
      ); // image upload dir
      const outputFile = path.join(
        __dirname,
        `embeddings`,
        `${rollno}-${name}.npy`
      );
      const saveEmbeddingsPromise = new Promise((resolve, reject) => {
        const pythonProcess = spawn("python", [
          "scripts/save_embeddings.py",
          inputDir,
          outputFile,
        ]);

        pythonProcess.stdout.on("data", (data) => {
          console.log(`stdout: ${data}`);
        });

        pythonProcess.stderr.on("data", (data) => {
          console.error(`stderr: ${data}`);
        });

        pythonProcess.on("close", (code) => {
          if (code === 0) {
            resolve(); // Resolve the promise when the Python script finishes successfully
          } else {
            reject(`Python process failed with code ${code}`);
          }
        });
      });

      // Wait for the save embeddings process to finish before sending the response
      await saveEmbeddingsPromise;

      res.status(200).json({ message: "All images processed", results });
      clearTargetFolder(req, res);
      return;
    } catch (err) {
      return res.status(400).json({ error: err.toString() });
    }
  }
);

app.post(
  "/api/generate-report",
  uploadVideoStorage.single("video"),
  async (req, res) => {
    console.log("  req.file", req.file);
    const uploadedVideoPath = path.join(
      __dirname,
      "uploads/test-video",
      req.file.originalname
    );
    const exelTemplatePath = path.join(
      __dirname,
      "template",
      `attendance_report.xlsx`
    );
    const outputReportPath = path.join(
      __dirname,
      "outputs",
      `attendance_report.xlsx`
    );
    const scriptPath = path.join(__dirname, "scripts/generate_report.py");

    const py = spawn("python", [
      scriptPath,
      req.file.originalname,
      exelTemplatePath,
    ]);

    py.stdout.on("data", (data) => {
      console.log(`📤 Python output: ${data}`);
    });

    py.stderr.on("data", (data) => {
      console.error(`⚠️ Python error: ${data}`);
    });

    py.on("close", (code) => {
      if (code === 0) {
        console.log("✅ Python script completed.");
        res.download(outputReportPath, "attendance_report.xlsx", (err) => {
          if (err) {
            console.error("❌ Error sending file:", err);
            res.status(500).send("Error sending file.");
          } else {
            // Optionally delete the report after sending
            // fs.unlink(outputReportPath, () => {});
          }
        });
      } else {
        console.error(`❌ Python process exited with code ${code}`);
        res.status(500).send("Python script failed.");
      }
      fs.unlink(uploadedVideoPath, (err) => {
        if (err) {
          console.error("❌ Failed to delete file:", err);
        } else {
          console.log("✅ File deleted successfully.");
        }
      });
    });
  }
);

app.post(
  "/api/attendence-image",
  uploadAttendenceImage.single("image"),
  async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No image uploaded" });
      }
      const imagePath = path.join(
        __dirname,
        "uploads/test-image",
        req.file.originalname
      );
      const outputPath = path.join(__dirname, "outputs/attendance.xlsx");
      const exelPath = path.join(__dirname, "template/attendance.xlsx");
      // Remove previous attendance file if exists
      if (fs.existsSync(outputPath)) fs.unlinkSync(outputPath);

      // Spawn Python script
      const scriptPath = path.join(__dirname, "scripts/attendance.py");

      const pythonProcess = spawn("python", [
        scriptPath,
        req.file.originalname,
        exelPath,
        // outputPath,
      ]);

      pythonProcess.stdout.on("data", (data) => {
        console.log("Python stdout:", data.toString());
      });

      pythonProcess.stderr.on("data", (data) => {
        console.error("Python stderr:", data.toString());
      });

      pythonProcess.on("close", (code) => {
        if (code !== 0) {
          return res.status(500).json({ error: "Python script failed" });
        }

        if (!fs.existsSync(outputPath)) {
          return res.status(500).json({ error: "Output file not found" });
        }

        // Send the generated file as a download
        res.download(outputPath, "attendance.xlsx", (err) => {
          if (err) {
            console.error("Download error:", err);
            res.status(500).end();
          }
        });
        // fs.unlink(outputPath, (err) => {
        //   if (err) {
        //     console.error("❌ Failed to delete file:", err);
        //   } else {
        //     console.log("✅ File deleted successfully.");
        //   }
        // });
        fs.unlink(imagePath, (err) => {
          if (err) {
            console.error("❌ Failed to delete file:", err);
          } else {
            console.log("✅ File deleted successfully.");
          }
        });
      });
    } catch (err) {
      console.error("Server error:", err);
      res.status(500).json({ error: "Internal server error" });
    }
  }
);
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
