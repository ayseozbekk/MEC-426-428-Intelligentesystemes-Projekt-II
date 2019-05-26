

const IMAGENET_CLASSES = {
   0: 'Apple Braeburn',
   1:'Apple Crimson Snow',
   2: 'Apple Golden 1',
   3: 'Apple Golden 2',
   4: 'Apple Golden 3',
   5: 'Apple Granny Smith',
   6: 'Apple Pink Lady',
   7: 'Apple Red 1',
   8: 'Apple Red 2',
   9: 'Apple Red 3',
   10: 'Apple Red Delicious',
   11: 'Apple Red Yellow 1',
   12: 'Apple Red Yellow 2',
   13:'Apricot',
   14: 'Avocado',
   15:'Avocado ripe',
   16: 'Banana',
   17: 'Banana Lady Finger',
   18: 'Banana Red',
   19:'Cactus fruit',
   20:'Cantaloupe 1',
   21: 'Cantaloupe 2',
   22:'Carambula',
   23: 'Cherry 1',
   24:'Cherry 2',
   25:'Cherry Rainier',
   26:'Cherry Wax Black',
   27:'Cherry Wax Red',
   28:'Cherry Wax Yellow',
   29:'Chestnut',
   30:'Clementine',
   31:'Cocos',
   32:'Dates',
   33:'Granadilla',
   34:'Grape Blue',
   35:'Grape Pink',
   36:'Grape White',
   37:'Grape White 2',
   38:'Grape White 3',
   39:'Grape White 4',
   40:'Grapefruit Pink',
   41:'Grapefruit White',
   42:'Guava',
   43:'Hazelnut',
   44:'Huckleberry',
   45:'Kaki',
   46:'Kiwi',
   47:'Kohlrabi',
   48:'Kumquats',
   49:'Lemon',
   50:'Lemon Meyer',
   51:'Limes',
   52:'Lychee',
   53:'Mandarine',
   54:'Mango',
   55:'Mangostan',
   56:'Maracuja',
   57:'Melon Piel de Sapo',
   58:'Mulberry',
   59:'Nectarine',
   60:'Orange',
   61:'Papaya',
   62:'Passion Fruit',
   63:'Peach',
   64:'Peach 2',
   65:'Peach Flat',
   66:'Pear',
   67:'Pear Abate',
   68:'Pear Kaiser',
   69:'Pear Monster',
   70:'Pear Red',
   71:'Pear Williams',
   72:'Pepino',
   73:'Pepper Green',
   74:'Pepper Red',
   75:'Pepper Yellow',
   76:'Physalis',
   77:'Physalis with Husk',
   78:'Pineapple',
   79:'Pineapple Mini',
   80:'Pitahaya Red',
   81:'Plum',
   82:'Plum 2',
   83:'Plum 3',
   84:'Pomegranate',
   85:'Pomelo Sweetie',
   86:'Quince',
   87:'Rambutan',
   88:'Raspberry',
   89:'Redcurrant',
   90:'Salak',
   91:'Strawberry',
   92:'Strawberry Wedge',
   93:'Tamarillo',
   94:'Tangelo',
   95:'Tomato 1',
   96:'Tomato 2',
   97:'Tomato 3',
   98:'Tomato 4',
   99:'Tomato Cherry Red',
   100:'Tomato Maroon',
   101:'Tomato Yellow',
   102:'Walnut'
  };
  

$(document).ready()
{
  $('.progress-bar').hide();
}
$("#image-selector").change(function(){
    let reader = new FileReader();

    reader.onload = function(){
        let dataURL = reader.result;
        $("#selected-image").attr("src",dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
});


$("#model-selector").change(function(){
    loadModel($("#model-selector").val());
    $('.progress-bar').show();
})


let model;
async function loadModel(name){
    model=await tf.loadModel(`http://localhost:8080/${name}/model.json`);
    $('.progress-bar').hide();
}


$("#predict-button").click(async function(){
    let image= $('#selected-image').get(0);
    let tensor = preprocessImage(image,$("#model-selector").val());

    let prediction = await model.predict(tensor).data();
    let top5=Array.from(prediction)
                .map(function(p,i){
    return {
        probability: p,
        className: IMAGENET_CLASSES[i]
    };
    }).sort(function(a,b){
        return b.probability-a.probability;
    }).slice(0,5);

$("#prediction-list").empty();
top5.forEach(function(p){
    $("#prediction-list").append(`<li>${p.className}:${p.probability.toFixed(6)}</li>`);
});

});


function preprocessImage(image,modelName)
{
    let tensor=tf.fromPixels(image)
    .resizeNearestNeighbor([224,224])
    .toFloat();//.sub(meanImageNetRGB)
          
    if(modelName==undefined)
    {
        return tensor.expandDims();
    }
    
    else if(modelName=="transfer")
    {
        let offset=tf.scalar(127.5);
        return tensor.sub(offset)
                    .div(offset)
                    .expandDims();
    }
    else
    {
        throw new Error("UnKnown Model error");
    }
}
