$(function () {
    $(".scroll-h").click(function () {
        $("html,body").animate({scrollTop: "0px"}, 800)
    });

    $(".scroll-b").click(function () {
        $("html,body").animate({scrollTop: $("footer").offset().top}, 800)
    });

    $(".qrshow").mouseover(function () {
        $(this).children(".qrurl-box").show()
    });
    $(".qrshow").mouseout(function () {
        $(this).children(".qrurl-box").hide()
    });

    $(".qrurl").mouseover(function () {
        var qrurl = window.location.href;
        if (!qrurl == "") {
            var qr = new QRious({
                element: document.getElementById('qrious'),
                size: 162,
                level: 'H',
                value: window.location.href
            });
        }
    });
});