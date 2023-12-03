///////////////////////////////////////////////////////////////////////////
//////////////////////// Input FCs of each branch in Pa structure /////////
///////////////////////////////////////////////////////////////////////////

var FC1 = S2_bands;
var FC2 = S2_indices;
var FC3 = L89_bands;
var FC4 = L89_indices;
var FC5 = S1_bands;

///////////////////////////////////////////////////////////////////////////
//////////////////////// PCA Function /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

var getPrincipalComponents = function(centered, scale, region) {
  var arrays = centered.toArray();
  var covar = arrays.reduceRegion({
    reducer: ee.Reducer.centeredCovariance(),
    geometry: region,
    scale: scale,
    maxPixels: 1e9
  });
  var covarArray = ee.Array(covar.get('array'));
  var eigens = covarArray.eigen();
  var eigenValues = eigens.slice(1, 0, 1);
  var eigenVectors = eigens.slice(1, 1);
  var arrayImage = arrays.toArray(1);
  var principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage);
  var sdImage = ee.Image(eigenValues.sqrt())
      .arrayProject([0]).arrayFlatten([getNewBandNames('sd')]);
  return principalComponents
      .arrayProject([0])
      .arrayFlatten([getNewBandNames('pc')])
      .divide(sdImage);
};

///////////////////////////////////////////////////////////////////////////
//////////////////////// Preparation of reference samples /////////////////
///////////////////////////////////////////////////////////////////////////


var bandnames_FC1 = FC1.bandNames();
var trainingSet_FC1 = FC1.sampleRegions({
collection:training,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

print(trainingSet_FC1)

var testingSet_FC1 = FC1.sampleRegions({
collection:test,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

var bandnames_FC2 = FC2.bandNames();

var trainingSet_FC2 = FC2.sampleRegions({
collection:training,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

var testingSet_FC2 = FC2.sampleRegions({
collection:test,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

var bandnames_FC3 = FC3.bandNames();

var trainingSet_FC3 = FC3.sampleRegions({
collection:training,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

var testingSet_FC3 = FC3.sampleRegions({
collection:test,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

var bandnames_FC4 = FC4.bandNames();

var trainingSet_FC4 = FC4.sampleRegions({
collection:training,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

var testingSet_FC4 = FC4.sampleRegions({
collection:test,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

var bandnames_FC5 = FC5.bandNames();

var trainingSet_FC5 = FC5.sampleRegions({
collection:training,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

var testingSet_FC5 = FC5.sampleRegions({
collection:test,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

// ///////////////////////////////////////////////////////////////////////////
// //////////////////////// Pa Structure ///////////////////////////////////
// ///////////////////////////////////////////////////////////////////////////


//// *** Note: Users must first find the optimal hyperparameters of each RF, which are:
//// * numberOfTrees, variablesPerSplit, and maxNodes.

var RF1 = ee.Classifier.smileRandomForest().setOutputMode('MULTIPROBABILITY');
var RF2 = ee.Classifier.smileRandomForest().setOutputMode('MULTIPROBABILITY');
var RF3 = ee.Classifier.smileRandomForest().setOutputMode('MULTIPROBABILITY');
var RF4 = ee.Classifier.smileRandomForest().setOutputMode('MULTIPROBABILITY');
var RF5 = ee.Classifier.smileRandomForest().setOutputMode('MULTIPROBABILITY');

var trained_RF1 = RF1.train(trainingSet_FC1, 'classvalue', bandnames_FC1);
var classified_FC1 = FC1.classify(trained_RF1);

var trained_RF2 = RF2.train(trainingSet_FC2, 'classvalue', bandnames_FC2);
var classified_FC2 = FC2.classify(trained_RF2);

var trained_RF3 = RF3.train(trainingSet_FC3, 'classvalue', bandnames_FC3);
var classified_FC3 = FC3.classify(trained_RF3);

var trained_RF4 = RF4.train(trainingSet_FC4, 'classvalue', bandnames_FC4);
var classified_FC4 = FC4.classify(trained_RF4);

var trained_RF5 = RF5.train(trainingSet_FC5, 'classvalue', bandnames_FC5);
var classified_FC5 = FC5.classify(trained_RF5);

// ///////////////////////////////////////////////////////////////////////////
// //////////////////////// Generation of Probability Maps ///////////////////
// ///////////////////////////////////////////////////////////////////////////

var FC1_C0 = classified_FC1.arrayGet(0);
var FC1_C1 = classified_FC1.arrayGet(1);
var FC1_C2 = classified_FC1.arrayGet(2);
var FC1_C3 = classified_FC1.arrayGet(3);
var FC1_C4 = classified_FC1.arrayGet(4);
var FC1_C5 = classified_FC1.arrayGet(5);
var FC1_C6 = classified_FC1.arrayGet(6);

var FC2_C0 = classified_FC2.arrayGet(0);
var FC2_C1 = classified_FC2.arrayGet(1);
var FC2_C2 = classified_FC2.arrayGet(2);
var FC2_C3 = classified_FC2.arrayGet(3);
var FC2_C4 = classified_FC2.arrayGet(4);
var FC2_C5 = classified_FC2.arrayGet(5);
var FC2_C6 = classified_FC2.arrayGet(6);

var FC3_C0 = classified_FC3.arrayGet(0);
var FC3_C1 = classified_FC3.arrayGet(1);
var FC3_C2 = classified_FC3.arrayGet(2);
var FC3_C3 = classified_FC3.arrayGet(3);
var FC3_C4 = classified_FC3.arrayGet(4);
var FC3_C5 = classified_FC3.arrayGet(5);
var FC3_C6 = classified_FC3.arrayGet(6);

var FC4_C0 = classified_FC4.arrayGet(0);
var FC4_C1 = classified_FC4.arrayGet(1);
var FC4_C2 = classified_FC4.arrayGet(2);
var FC4_C3 = classified_FC4.arrayGet(3);
var FC4_C4 = classified_FC4.arrayGet(4);
var FC4_C5 = classified_FC4.arrayGet(5);
var FC4_C6 = classified_FC4.arrayGet(6);

var FC5_C0 = classified_FC5.arrayGet(0);
var FC5_C1 = classified_FC5.arrayGet(1);
var FC5_C2 = classified_FC5.arrayGet(2);
var FC5_C3 = classified_FC5.arrayGet(3);
var FC5_C4 = classified_FC5.arrayGet(4);
var FC5_C5 = classified_FC5.arrayGet(5);
var FC5_C6 = classified_FC5.arrayGet(6);

// ///////////////////////////////////////////////////////////////////////////
// ///////////////////// Class Wise Arrangement /////////////////////////
// ///////////////////////////////////////////////////////////////////////////

var PM_C0 = FC1_C0.addBands(FC2_C0).addBands(FC3_C0).addBands(FC4_C0).addBands(FC5_C0);
var PM_C1 = FC1_C1.addBands(FC2_C1).addBands(FC3_C1).addBands(FC4_C1).addBands(FC5_C1);
var PM_C2 = FC1_C2.addBands(FC2_C2).addBands(FC3_C2).addBands(FC4_C2).addBands(FC5_C2);
var PM_C3 = FC1_C3.addBands(FC2_C3).addBands(FC3_C3).addBands(FC4_C3).addBands(FC5_C3);
var PM_C4 = FC1_C4.addBands(FC2_C4).addBands(FC3_C4).addBands(FC4_C4).addBands(FC5_C4);
var PM_C5 = FC1_C5.addBands(FC2_C5).addBands(FC3_C5).addBands(FC4_C5).addBands(FC5_C5);
var PM_C6 = FC1_C6.addBands(FC2_C6).addBands(FC3_C6).addBands(FC4_C6).addBands(FC5_C6);

///////////////////////////////////////////////////////////////////////////
//////////////////////////////// Apply PCA on PMs /////////////////////////
///////////////////////////////////////////////////////////////////////////

var bandNames_C0 = PM_C0.bandNames();

var getNewBandNames = function(prefix) {
  var seq = ee.List.sequence(1, bandNames_C0.length());
  return seq.map(function(b) {
    return ee.String(prefix).cat(ee.Number(b).int());
  });
};

var PCA_PM_C0 = getPrincipalComponents(PM_C0,30,study_area).clip(study_area);
var PCA_PM_C1 = getPrincipalComponents(PM_C1,30,study_area).clip(study_area);
var PCA_PM_C2 = getPrincipalComponents(PM_C2,30,study_area).clip(study_area);
var PCA_PM_C3 = getPrincipalComponents(PM_C3,30,study_area).clip(study_area);
var PCA_PM_C4 = getPrincipalComponents(PM_C4,30,study_area).clip(study_area);
var PCA_PM_C5 = getPrincipalComponents(PM_C5,30,study_area).clip(study_area);
var PCA_PM_C6 = getPrincipalComponents(PM_C6,30,study_area).clip(study_area);

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Stacking of the top Components //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

var com_1 = PCA_PM_C0.select(0).addBands(PCA_PM_C1.select(0)).addBands(PCA_PM_C2.select(0))
.addBands(PCA_PM_C3.select(0)).addBands(PCA_PM_C4.select(0)).addBands(PCA_PM_C5.select(0)).addBands(PCA_PM_C6.select(0));

// ////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////// Ca structure for final classification ///////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////

var trainingSet_com_1 = com_1.sampleRegions({
collection:training,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

var testingSet_com_1 = com_1.sampleRegions({
collection:test,
properties: ["classvalue"],
scale: 30,
tileScale:16,
geometries:true
});

var com_1_bandnames = com_1.bandNames();

//// *** Note: Users must first find the optimal hyperparameters of each RF, which are:
//// * numberOfTrees, variablesPerSplit, and maxNodes.

var Meta_model = ee.Classifier.smileRandomForest();
var Meta_model = Meta_model.train(trainingSet_com_1, 'classvalue', com_1_bandnames);
var classified_com_1 = com_1.classify(Meta_model);

var imageVisParam = ["ffe5b4","9f6149","4eff8e","faff06","9e76ff","ff0000","00a10d"];
Map.addLayer(classified_com_1, {palette: imageVisParam, min:0,max:6}, 'classified_com_1',false);

//////////////////// Validation ///////////////////////
var test_com_1 = testingSet_com_1.classify(Meta_model);
var confusionMatrix_com_1 = test_com_1.errorMatrix('classvalue', 'classification');
print("Overall accuracy_com_1", confusionMatrix_com_1.accuracy());
print('Kappa statistic_com_1', confusionMatrix_com_1.kappa());