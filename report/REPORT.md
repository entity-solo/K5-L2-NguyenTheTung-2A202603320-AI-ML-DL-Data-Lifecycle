# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** ___

**Runtime Colab:** CPU / GPU *(xóa cái không dùng)*

**Python / PyTorch / Ultralytics:** ___ / ___ / ___

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` *(chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)*

**Thay đổi so với notebook nguồn:** Không *(nếu có thay đổi, mô tả rõ ở đây)*

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

---

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số "tin" của model; không phải điểm chất lượng nhãn người |

---

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

**1.** Chép record hạng 1:
- `class_id` = 468
- `class_name` = "cab"
- `rank` = 1
- `score` = 0.510915
- `taxonomy_name` = "ImageNet-1K"

**2.** Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao?

> ___ Record này nói về cả bức ảnh vì đây là tác vụ phân loại (classification), model nhận ảnh rồi trả về dạng JSON các candicate tiềm năng cho việc phân loại, không có bbox hay polygon.

**3.** Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

> ___ Danh sách các tên lớp đi kèm với taxonomy/checkpoint, checkpoint yolo11n-cls.pt được huấn luyện trên taxonomy ImageNet-1K (1000 lớp định sẵn)

**4.** Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

> ___ `taxonomy_name`: danh sách lớp được định nghĩa để train, `class_name`: tên của lớp đó để đọc nhanh, `class_id`: mã định danh của lớp đó trong danh sách lớp (taxonomy), 3 yếu tố trên giúp cho việc truy vết chính xác

**5.** Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

> ___ Guideline cần nói rõ về tiêu chí về việc gán nhãn (VD: Vật chiếm diện tích lớn nhất, chủ thể chính, thứ tự ưu tiên) để việc gán nhãn được nhất quán giữa những người gán nhãn

**6.** Vì sao không được lấy `score` của model làm ground truth?

> ___ Vì score là mức độ tự tin của model và model có thể sai, trong khi ground truth là sự thật tuyệt đối

---

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

**1.** Chép một record:
- `class_name` = "person"
- `score` = 0.912625
- `bbox_xyxy` = [385.33, 69.24, 498.92, 348.92]
- `bbox_width` = 113.58
- `bbox_height` = 279.68

**2.** Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

> ___ Góc trên-trái ở [385.33, 69.24], góc dưới-phải ở [498.92, 348.92], rộng 113.58 pixel, cao 279.68 pixel

**3.** Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

| Ngưỡng | Số prediction |
| --- | --- |
| 0.20 | 11 |
| 0.60 | 6 |

**4.** Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

> ___ Khi hạ ngưỡng, người reviewer xóa box sai, kiểm tra nhãn lớp từng box, và xử lý những box trùng lặp

**5.** Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

> ___ Bounding box phải bao sát vật thể nhất, các cạnh chạm đúng biên ngoài cùng của vật thể

**6.** Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

> Vật: Cốc
>
> Chọn: guideline
>
> Lý do: Vật bị che khuất khá phổ biến trong ảnh, nếu vật bị che khuất vẫn có thể nhận diện được, vẫn gán box bao sát những phần nhìn thấy được của vật đó

---

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

**1.** Chép một record:
- `instance_id` = "kitchen-009"
- `class_name` = "person"
- `score` = 0.431438
- `polygon_point_count` = 65
- 3–5 điểm đầu của `polygon_xy` = [1.0, 264.0], [1.0, 309.0], [5.0, 309.0], [6.0, 308.0], [15.0, 308.0]

**2.** So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

> ___ Polygon bao sát vật thể hơn box, từ đó cho thấy hình dạng vật thể rõ hơn, trong khi box để lại khá nhiều khoảng trống không phải là một phần của thực thể

**3.** `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

> ___ `instance_id` dùng để phân biệt từng vật thể trong cùng một ảnh, nó không phải class ID vì class ID chỉ loại vật thể trong khi đó có thể có nhiều vật cùng lại vật thể trong một ảnh, nó không phải tracking ID do tracking ID dùng để theo dõi một vật qua nhiều frame video, còn `instance_id` chỉ có hiệu lực trong một ảnh

**4.** Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

> ___ Vẽ polygon bám sát sao cho từng pixel mà polygon bao trọn thuộc về vật thể

**5.** Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

> Chỗ khó: Hai vật chạm nhau
>
> Chọn: guideline
>
> Lý do: Hai vật chạm nhau là tình huống phổ biến trong ảnh, annotator cần đảm bảo từng pixel polygon bao trọn thuộc về vật thể đó

---

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất cho cả ảnh | Ảnh có nhiều vật nhiều chủ đề | Nhìn ảnh, làm theo guideline chọn một lớp phù hợp nhất | Kiểm tra nhãn có khớp theo guideline, có nhất quán trên nhiều ảnh tương tự không? |
| Phát hiện vật thể | Một vật một bounding box | Hạ ngưỡng xuất hiện nhiều box sai trùng, vật che khuất cắt mép khó quyết định gán box | Vẽ box bao sát mép ngoài cùng của vật | Kiểm tra box có sát vật, có box thừa thiếu không? Nhãn lớp có đúng không? |
| Instance segmentation | Một vật một polygon | Hai vật chạm nhau biên khó tách, vật trong suốt biên mờ, model có thể tạo polygon lấn sang nền | Kiểm tra xem từng pixel trong polygon có thuộc vật đó không? | Kiểm tra polygon có sát biên không? Có bị gán nhầm không? |

---

## 5. An toàn dữ liệu

**1.** Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

> ___ Không ghi họ tên, MSSV, email, số điện thoại hoặc bất kỳ thông tin cá nhân nào vào báo cáo, code, hay output — chỉ đưa thông tin định danh vào tên repository.

**2.** Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

> ___ Dừng ngay, không xử lý tiếp, và báo cho Supervisor để được hướng dẫn.

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
