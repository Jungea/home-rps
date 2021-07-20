$(document).ready(function() {
    result_string = ['Rock', 'Paper', 'Scissors']  //예측값에 따른 이름
    computer_hand = ['paper', 'scissors', 'rock']  //예측값에 따른 컴퓨터 낼 것

    setInterval(getlabel, 1000); // 예측값 1초에 한번씩 불러옴

    //예측값 호출
    function getlabel() {

	    $.ajax({
	        url: "/get_label",
            method: "GET",
            dataType: "json",

        }).done(function(r) {

            var result = r.result;
            $('#result').text(result_string[result]);
            $('#computer_hand').attr('src', '/static/image/'+computer_hand[result]+'.png')

        console.log("ajax-getLabel-success", result);

      }).fail(function() {
            console.log("ajax-getLabel-fail");
      });

	}
})