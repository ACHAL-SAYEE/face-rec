const mongoose = require("mongoose");

const UserSchema = new mongoose.Schema({
  userId: String,
  email: String,
  password: String,
  role: String,
  firstName: { type: String, default: null },
  surName: { type: String, default: null },
  gender: { type: String, default: null },
  dob: { type: String, default: null },
  // department: { type: String, default: null },
  info: { type: Object },
  friends: { type: Array, default: [] },
  isVerified: { type: Boolean, default: false },
  classRooms: { type: [String], default: [] },
});

// const Student = mongoose.model("student", StudentSchema);
const User = mongoose.model("user", UserSchema);

exports.User = User;
