const functions = require("firebase-functions");
const fetch = require("cross-fetch");

//수정 시 발동 안 함 user폴더 자체가 다시 생기지만 않게 하면 됨
exports.onNewTargetUpload = functions.database
  .ref("/animalList/{targetId}")
  .onCreate(async (snapshot, context) => {
    const targetId = context.params.targetId;

    console.log(`New target uploaded: target${targetId}`);

    //************나중에 url 수정*****************
    try {
      const response = await fetch("https://divi.kro.kr/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targetId: targetId }),
      });

      console.log("Fetch 요청 보냄");

      if (!response.ok) {
        throw new Error(`Flask 서버 오류: ${response.statusText}`);
      }
      const result = await response.json();
      console.log("Flask API response:", result);
    } catch (error) {
      console.error("Error calling Flask API:", error);
    }

    return null;
  });
