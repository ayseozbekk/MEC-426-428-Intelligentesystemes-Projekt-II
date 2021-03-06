//Defining the Objects
let express = require("express");
let app = express();

app.use(function(req, res, next){
    console.log(`${new Date()} - ${req.method} reqest for ${req.url}`);
    next();
});

app.use(express.static("../client"));

//Listening to Requests from Client
app.listen(8080, function(){
    console.log("Serving at 8080")
});
