# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603857
- **Họ và tên:** Nguyễn Lê Trọng Mạnh
- **SEED** (in ở ô đầu notebook): 5

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

## 3. Dự đoán và thực tế

ml_f1: tôi đoán F1 ≈ 0.88, recall cao hơn precision. Thực tế F1 = 0.778, precision 0.984, recall 0.643. Lệch vì mô hình chỉ chặn khi rất chắc, từ ham phổ biến (“i”, “my”, “you”) kéo điểm xuống nên bỏ sót hơn một phần ba spam.
emb_search: tôi đoán kết quả gần nhất là ham. Thực tế cả 6 kết quả đều là spam (cosine 0.53–0.56). Lệch vì embedding hiểu nghĩa (“complimentary holiday” gần “Tenerife Holiday award”) dù không có từ khoá quen.
dl_relu: tôi nghĩ bỏ ReLU sẽ làm mô hình kém đi. Thực tế F1 không ReLU vẫn 0.878, bằng MLP đầy đủ. Lệch vì hai lớp Linear liên tiếp tương đương một lớp tuyến tính, nên khoảng cách 0.778 → 0.878 chủ yếu do cách huấn luyện (Adam, minibatch), không phải phi tuyến.

## 4. Phân tích 50 case khó

Tụt nhiều nhất: BoW + LogReg và BoW + MLP, chỉ đúng khoảng 53% và 51% (F1 0.350 và 0.372, so với F1 test thường 0.778 và 0.878). Nhóm yếu nhất: Thao túng/lừa đảo (10%), Spam giọng tự nhiên (20%), Né từ khoá (17% với LogReg, 0% với MLP).
Case 1 (né từ khoá): C0ngr4tulat10ns! Y0u h4ve w0n .... Regex tách token làm các từ viết thay số không nằm trong vocab, nên BoW mất toàn bộ tín hiệu spam.
Case 2 (thao túng): “Hi Mum, I dropped my phone in the toilet, this is my new number...”. Không có từ khoá spam nào; lừa đảo nằm ở ngữ cảnh và ý định, mà BoW chỉ đếm từ.

## 5. Thử thách cá nhân

Đề #5 (SEED = 5464, 5464 % 6 = 4): viết 4 tin (2 spam né từ khoá, 2 ham chứa free/win), so BoW và Embedding+LogReg.
Cách làm: tự viết 4 tin, dự đoán bằng predict_proba(vectorize(), w, b) cho BoW và predict_proba((emb_model.encode() - mu)/sd, w_e, b_e) cho Embedding.
Kết quả: BoW đúng 2/4, Embedding đúng 4/4. Tin spam 1 và 2: BoW P = 0.196 và 0.479 (sai), Embedding P = 0.684 và 0.861 (đúng). Tin ham 3 và 4: cả hai đúng.
Giải thích: BoW mất tín hiệu vì cl@im, c.a.s.h bị tách vụn hoặc không có trong vocab. Embedding đọc nghĩa cả câu. Với tin ham, BoW vẫn đúng ngược dự đoán của tôi vì các từ ham thông dụng kéo điểm xuống, lấn át trọng số của “free” và “win”.

## 6. Phương án cho sản phẩm thật

Chọn: BoW + MLP làm tầng chính, kèm bước chuẩn hoá đầu vào, và LLM nhỏ chỉ xử lý tin lưỡng lự (p ∈ [0.3, 0.7]).
Số liệu: chỉ hai mô hình BoW đạt p95 < 50 ms (0.33 và 3.57 ms), ba cách Encoder trên 60 ms (CPU). BoW + MLP có F1 0.878, cao hơn LogReg 0.778. Dùng LLM cho tất cả 10 triệu tin/ngày tốn khoảng $105/ngày (≈ $3.150/tháng, tính tay); nếu chỉ 5% tin lưỡng lự (giả định, cần đo lại) thì khoảng $5/ngày.
Chấp nhận đánh đổi: mất một phần chất lượng so với dùng LLM hoàn toàn, và rủi ro BoW sai mà vẫn tự tin (tin 1 thử thách P = 0.196 sẽ không lên tầng 2). Giảm bằng chuẩn hoá đầu vào, mở rộng vùng lưỡng lự, thêm quy tắc rẻ (link lạ, số 09...).

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời / viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| 1 (ML: công thức) | Hỏi công thức Sigmoid, gradient descent cho Logistic Regression, Precision/Recall/F1 | Giải thích σ(z), grad_w = Xᵀ(p − y)/n, cách tránh chia cho 0 khi tính F1 | Tự viết `vectorize`, `sigmoid`, `predict_proba`, `train_logreg`, `metrics`; chạy pass `_run_check` 1.1–1.4 |
| 2 (DL: MLP) | Hỏi số tham số MLP, điều gì xảy ra khi bỏ `ReLU`, lỗi lệch shape khi tính loss | Hai Linear liên tiếp tương đương một Linear; dùng `.squeeze(1)` | Tự viết `build_mlp`, `train_mlp`; kiểm chứng 128,129 tham số và F1 không ReLU = 0.878 |
| 3 (Embedding: cosine) | Hỏi cách chuẩn hoá vector để tính cosine bằng nhân ma trận | Chia mỗi vector cho chuẩn L2 | Tự viết `cosine_similarity`, `top_k_similar`, `knn_predict`; chạy pass `_run_check` 3.1–3.3 |
| 4 (TODO 4.1) | Hoàn thành `call_openai` | Hàm gọi Chat Completions kèm `**provider_kwargs`, đo latency | Chạy ô kiểm tra mock và gọi API thật thành công |
| 4 (TODO 4.2–4.3) | Viết `build_classification_prompt`, `parse_label` | Prompt zero/few-shot; parse bắt "not spam" trước "spam" | Chạy `_run_check` 4.2–4.3 |
| 4 (TODO 4.4) | Hoàn thành `llm_classify_batch` | Vòng lặp gọi API có retry, parse, ước tính token và chi phí | Phát hiện chi phí ra 0 do tên model không có trong bảng giá, rồi sửa |
| 5 (thử thách đề #5) | Sửa lỗi `text_to_bow` và gợi ý code | Code so BoW với Embedding+LogReg trên 4 tin | Đổi tên cho khớp notebook, tự chạy và đọc kết quả. Lưu ý: đề ghi "không dùng AI" |
| 5 (tính tay) | Hướng dẫn các bước tính tay | Công thức và các bước thay số | Tính lại trên giấy, kiểm tra khớp số liệu |
| 5 (stress test, 5 case) | Giải thích câu hỏi phân tích và gợi ý case | Phân tích điểm yếu từng model; 5 case `MY_CASES` | Đối chiếu với heatmap thực tế, viết lại bằng lời của mình |
| Rà soát | Tìm và phân tích lỗi sai | Danh sách lỗi: nhận định lệch output, chi phí LLM bằng 0 | Đối chiếu với output, sửa câu trả lời và `HAND_CALC` |
| 5 (báo cáo, phản ánh) | Soạn bản nháp mục 2–7 | Bản nháp kèm số liệu | Thay bằng số thật, viết lại bằng lời của mình |

**Điều đã học:** mô hình phức tạp hơn chưa chắc tốt hơn (bỏ ReLU, F1 vẫn 0.878); điểm test thường dễ lạc quan (BoW F1 0.778 nhưng chỉ khoảng 0.35 trên case khó); chọn phương án phải nhìn cùng lúc chất lượng, độ trễ và chi phí.
