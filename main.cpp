#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

bool is_image_file(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    return ext == ".jpg" ||
           ext == ".jpeg" ||
           ext == ".png" ||
           ext == ".bmp" ||
           ext == ".webp";
}

torch::Tensor preprocess_image(
    const std::string& image_path,
    int image_size,
    torch::Device device
) {
    // 1. Read image with OpenCV.
    // OpenCV loads images as BGR by default.
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        throw std::runtime_error("Could not read image: " + image_path);
    }

    // 2. Convert BGR to RGB.
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // 3. Resize to model input size.
    cv::resize(image, image, cv::Size(image_size, image_size));

    // 4. Convert uint8 [0, 255] to float32 [0, 1].
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    // 5. Create tensor from OpenCV image.
    // Shape is H x W x C.
    torch::Tensor tensor = torch::from_blob(
        image.data,
        {image.rows, image.cols, 3},
        torch::kFloat32
    );

    // 6. Convert HWC to CHW.
    tensor = tensor.permute({2, 0, 1});

    // 7. Add batch dimension: CHW -> NCHW.
    tensor = tensor.unsqueeze(0);

    // 8. Normalize.
    // Replace these with the mean/std used during training.
    tensor = tensor.clone(); // Own the memory before OpenCV Mat goes out of scope.

    torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406})
                             .view({1, 3, 1, 1});
    torch::Tensor std = torch::tensor({0.229, 0.224, 0.225})
                            .view({1, 3, 1, 1});

    tensor = (tensor - mean) / std;

    // 9. Move to CPU or CUDA.
    tensor = tensor.to(device);

    return tensor;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./cnn_infer <model_path.pt> <image_folder>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_folder = argv[2];

    const int image_size = 224;

    torch::Device device(torch::kCPU);

    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using CUDA\n";
    } else {
        std::cout << "Using CPU\n";
    }

    try {
        // Load TorchScript model.
        torch::jit::script::Module model = torch::jit::load(model_path);
        model.to(device);
        model.eval();

        // Disable autograd for inference.
        torch::NoGradGuard no_grad;

        for (const auto& entry : fs::directory_iterator(image_folder)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            fs::path image_path = entry.path();

            if (!is_image_file(image_path)) {
                continue;
            }

            try {
                torch::Tensor input = preprocess_image(
                    image_path.string(),
                    image_size,
                    device
                );

                // Forward pass.
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input);

                torch::Tensor output = model.forward(inputs).toTensor();

                // Example for classification:
                // output shape: [1, num_classes]
                torch::Tensor probabilities = torch::softmax(output, 1);
                torch::Tensor predicted_class = torch::argmax(probabilities, 1);

                int class_id = predicted_class.item<int>();
                float confidence = probabilities[0][class_id].item<float>();

                std::cout << image_path.filename().string()
                          << " -> class: " << class_id
                          << ", confidence: " << confidence
                          << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Failed on image "
                          << image_path
                          << ": "
                          << e.what()
                          << "\n";
            }
        }
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
