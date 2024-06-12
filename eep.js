const express = require("express");
const app = express();
const path = require("path");
const port = 8080;

app.set("view engine","ejs");
app.set("views",path.join(__dirname,"/views"))

app.get("/",(req,res)=> {
    res.render("home.ejs");
});
app.get("/ig/:username",(req,res)=>{
    const followers = ["adam","bob","steve","abc"];
    let {username} = req.params;
    res.render("instagram.ejs",{username,followers})
})

app.get("/hello",(req,res)=> {
    res.send("hello");
});
app.get("/rolldice",(req,res)=> {
    let diceval = Math.floor(Math.random()*6)+1;
    res.render("rolldice.ejs",{diceval});
});

app.listen(port,()=>{
    console.log(`listening on port ${port}`);
});
