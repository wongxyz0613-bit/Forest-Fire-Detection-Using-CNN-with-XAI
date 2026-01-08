$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#address').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Log the entire response for debugging
                console.log(data); // Debugging: log the response

                // Hide loader and show result container
                $('.loader').hide();
                $('#result').fadeIn(600);

                // Display prediction text
                $('#result').text('Result: ' + data.prediction); // Access prediction from JSON

                // If SHAP image exists in the response, append it below the result text
                if (data.shap_image) {
                    $('#result').append(
                        '<img src="data:image/png;base64,' + data.shap_image + '" alt="SHAP Explanation" style="max-width: 500px;">'
                    );
                }

                // Show address input if fire is detected based on prediction text
                if (data.prediction.includes("Fire-detected")) {
                    $('#address').fadeIn(600);
                } else {
                    $('#address').hide();
                }
            },
            error: function (xhr, status, error) {
                $('.loader').hide(); // Hide loader on error
                console.error('Error occurred:', error); // Log error
                alert('An error occurred while processing your request. Please try again.'); // Alert user
            }
        });
    });



});
