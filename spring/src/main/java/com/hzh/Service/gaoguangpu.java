package com.hzh.Service;

import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.core.io.ByteArrayResource;

import java.util.HashMap;
import java.util.Map;

@Service
public class gaoguangpu {

    // 你的 Python 服务地址
    private static final String AI_SERVICE_URL = "http://localhost:8000/predict";

    public Map<String, Object> predict(MultipartFile file) {
        Map<String, Object> response = new HashMap<>();

        try {
            // 1. 把文件转成字节数组
            byte[] fileBytes = file.getBytes();

            // 2. 构造 form-data 请求（匹配 Python 端的 UploadFile）
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            // 用 MultiValueMap 模拟表单上传
            org.springframework.util.MultiValueMap<String, Object> body =
                    new org.springframework.util.LinkedMultiValueMap<>();

            body.add("file", new ByteArrayResource(fileBytes) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            });

            HttpEntity<org.springframework.util.MultiValueMap<String, Object>> requestEntity =
                    new HttpEntity<>(body, headers);

            // 3. 调用 Python 服务
            RestTemplate restTemplate = new RestTemplate();
            Map<String, Object> aiResult = restTemplate.postForObject(
                    AI_SERVICE_URL,
                    requestEntity,
                    Map.class
            );

            // 4. 封装返回结果
            response.put("code", 200);
            response.put("message", "success");
            response.put("data", aiResult);

        } catch (Exception e) {
            e.printStackTrace();
            response.put("code", 500);
            response.put("message", "AI 服务调用失败: " + e.getMessage());
        }

        return response;
    }
}
