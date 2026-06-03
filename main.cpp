#include <iostream>
#include <filesystem>
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>
//15: Momentos invariantes de Hu (7)
//16: Área, perímetro, longitud, centroide, orientación
//18: Factor circularidad / compactación
//22: Radii
namespace fs = std::filesystem;
// Carga la matriz binaria donde la imagen ya fue separada del fondo por Otsu
cv::Mat cargarImagenBinaria(const std::string& ruta) {
    return cv::imread(ruta, cv::IMREAD_GRAYSCALE);
}

//calcula todas las metricas de forma y momentos invariantes
void ejecutarMotorGeometrico(const cv::Mat& binaria) {
    std::cout << "[Modulo] Iniciando Analisis Geometrico e Invariante..." << std::endl;

    double area = 0.0;
    double m10 = 0.0;
    double m01 = 0.0;
    // Calculo de los momentos espaciales crudos
    // El momento m00 equivale al area total (suma de pixeles blancos)
    // Los momentos m10 y m01 acumulan la posicion de los objetos en los ejes X y Y
    for (int y = 0; y < binaria.rows; ++y) {
        for (int x = 0; x < binaria.cols; ++x) {
            if (binaria.at<uchar>(y, x) >= 127) {
                area += 1.0;
                m10 += x;
                m01 += y;
            }
        }
    }

    if (area == 0.0) {
        std::cout << "Error: No se encontraron regiones segmentadas." << std::endl;
        return;
    }
    //centroide
    double cx = m10 / area;
    double cy = m01 / area;
    // Declaracion de las variables para los momentos centrales (mu)
    // Estos momentos trasladan el origen del plano cartesiano hacia el centroide
    // asi las mediciones son invariantes a la traslacion
    double mu20 = 0.0, mu02 = 0.0, mu11 = 0.0;
    double mu30 = 0.0, mu03 = 0.0, mu21 = 0.0, mu12 = 0.0;

    double perimetro = 0.0;
    double radioMaximo = 0.0;
    double radioMinimo = std::numeric_limits<double>::max();

    int vecinosX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int vecinosY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    cv::Mat visualizacion;
    cv::cvtColor(binaria, visualizacion, cv::COLOR_GRAY2BGR);
    visualizacion *= 0.3;

    for (int y = 0; y < binaria.rows; ++y) {
        for (int x = 0; x < binaria.cols; ++x) {
            if (binaria.at<uchar>(y, x) >= 127) {
                // Calculo de la distancia de cada pixel respecto al centroide
                double dx = x - cx;
                double dy = y - cy;
                // Construccion de los momentos centrales de segundo y tercer orden
                mu20 += std::pow(dx, 2);
                mu02 += std::pow(dy, 2);
                mu11 += dx * dy;

                mu30 += std::pow(dx, 3);
                mu03 += std::pow(dy, 3);
                mu21 += std::pow(dx, 2) * dy;
                mu12 += dx * std::pow(dy, 2);
                // Para identificar si el pixel actual es frontera
                bool esFrontera = false;
                for (int i = 0; i < 8; ++i) {
                    int nx = x + vecinosX[i];
                    int ny = y + vecinosY[i];
                    if (nx < 0 || nx >= binaria.cols || ny < 0 || ny >= binaria.rows || binaria.at<uchar>(ny, nx) < 127) {
                        esFrontera = true;
                        break;
                    }
                }
                // Si es frontera, aporta al perimetro y se miden sus radios
                if (esFrontera) {
                    perimetro += 1.0;
                    visualizacion.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                    double distancia = std::sqrt(dx * dx + dy * dy);
                    if (distancia > radioMaximo) radioMaximo = distancia;
                    if (distancia < radioMinimo) radioMinimo = distancia;
                }
            }
        }
    }

    double circularidad = (perimetro > 0.0) ? (4.0 * CV_PI * area) / (perimetro * perimetro) : 0.0;
    double relacionRadios = (radioMaximo > 0.0) ? (radioMinimo / radioMaximo) : 0.0;
    // Division de los momentos de segundo orden entre el area para extraer la varianza espacial
    double u20 = mu20 / area;
    double u02 = mu02 / area;
    double u11 = mu11 / area;
    // Calculo del mayor valor propio (Eigenvalue) de la Matriz de Covarianza
    // extrae la longitud y orientacion del eje principal de la forma
    double discriminante = std::sqrt(4.0 * u11 * u11 + std::pow(u20 - u02, 2));
    double lambda1 = (u20 + u02) / 2.0 + discriminante / 2.0;
    double longitudEjeMayor = 4.0 * std::sqrt(lambda1);
    // Calculo del arco tangente para aislar el angulo de inclinacion
    double orientacionRad = 0.5 * std::atan2(2.0 * u11, u20 - u02);
    double orientacionGrados = orientacionRad * 180.0 / CV_PI;
    // Calcular los momentos centrales normalizados (eta)
    // Para que las mediciones sean invariantes al escalamiento
    auto eta = [&](double mu, int p, int q) {
        double gamma = (p + q) / 2.0 + 1.0;
        return mu / std::pow(area, gamma);
    };

    double n20 = eta(mu20, 2, 0);
    double n02 = eta(mu02, 0, 2);
    double n11 = eta(mu11, 1, 1);
    double n30 = eta(mu30, 3, 0);
    double n03 = eta(mu03, 0, 3);
    double n21 = eta(mu21, 2, 1);
    double n12 = eta(mu12, 1, 2);
    // Expansion polinomica de los 7 Momentos de Hu
    // combinan los momentos normalizados para crear las firmas
    // que son invariantes a la rotacion, traslacion y escala
    double h[7];
    h[0] = n20 + n02;
    h[1] = std::pow(n20 - n02, 2) + 4.0 * std::pow(n11, 2);
    h[2] = std::pow(n30 - 3.0 * n12, 2) + std::pow(3.0 * n21 - n03, 2);
    h[3] = std::pow(n30 + n12, 2) + std::pow(n21 + n03, 2);
    h[4] = (n30 - 3.0 * n12) * (n30 + n12) * (std::pow(n30 + n12, 2) - 3.0 * std::pow(n21 + n03, 2)) +
           (3.0 * n21 - n03) * (n21 + n03) * (3.0 * std::pow(n30 + n12, 2) - std::pow(n21 + n03, 2));
    h[5] = (n20 - n02) * (std::pow(n30 + n12, 2) - std::pow(n21 + n03, 2)) +
           4.0 * n11 * (n30 + n12) * (n21 + n03);
    h[6] = (3.0 * n21 - n03) * (n30 + n12) * (std::pow(n30 + n12, 2) - 3.0 * std::pow(n21 + n03, 2)) -
           (n30 - 3.0 * n12) * (n21 + n03) * (3.0 * std::pow(n30 + n12, 2) - std::pow(n21 + n03, 2));

    cv::circle(visualizacion, cv::Point(std::round(cx), std::round(cy)), 3, cv::Scalar(0, 255, 0), -1);
    cv::circle(visualizacion, cv::Point(std::round(cx), std::round(cy)), std::round(radioMaximo), cv::Scalar(0, 0, 255), 1);
    cv::circle(visualizacion, cv::Point(std::round(cx), std::round(cy)), std::round(radioMinimo), cv::Scalar(255, 0, 0), 1);

    std::cout << "Area: " << area << " px^2" << std::endl;
    std::cout << "Perimetro: " << perimetro << " px" << std::endl;
    std::cout << "Centroide (x,y): (" << cx << ", " << cy << ")" << std::endl;
    std::cout << "Eje Mayor (Longitud): " << longitudEjeMayor << " px" << std::endl;
    std::cout << "Orientacion Horizontal: " << orientacionGrados << " grados" << std::endl;
    std::cout << "Radio Minimo: " << radioMinimo << " px" << std::endl;
    std::cout << "Radio Maximo: " << radioMaximo << " px" << std::endl;
    std::cout << "Relacion (Rmin/Rmax): " << relacionRadios << std::endl;
    std::cout << "Factor de Circularidad: " << circularidad << std::endl;

    std::cout << "\n--- MOMENTOS INVARIANTES DE HU ---" << std::endl;
    for (int i = 0; i < 7; i++) {
        std::cout << "Hu[" << i + 1 << "]: " << h[i] << std::endl;
    }

    cv::imwrite("d_geometria_output.jpg", visualizacion);
    std::cout << "\n-> Archivo visual generado: 'd_geometria_output.jpg'" << std::endl;
}

int main() {

    std::string inputPath = "b_otsu_output.jpg";
    if (!fs::exists(inputPath)) {
        std::cout << "Error: No se encontro 'b_otsu_output.jpg'." << std::endl;
        return -1;
    }

    cv::Mat binaria = cargarImagenBinaria(inputPath);
    if (binaria.empty()) return -1;

    ejecutarMotorGeometrico(binaria);

    return 0;
}
