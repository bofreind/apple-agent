package com.hzh.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/cifar")
public class CifarController {

    private static final String PYTHON_SERVICE_URL = "http://localhost:8000/describe";

    @PostMapping("/describe")
    public Map<String, Object> describeImage(@RequestParam("file") MultipartFile file) {
        Map<String, Object> response = new HashMap<>();

        try {
            // 1. 构造 multipart/form-data 请求
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            // 用 MultiValueMap 模拟表单上传
            org.springframework.util.MultiValueMap<String, Object> body =
                    new org.springframework.util.LinkedMultiValueMap<>();

            body.add("file", new org.springframework.core.io.ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            });

            HttpEntity<org.springframework.util.MultiValueMap<String, Object>> requestEntity =
                    new HttpEntity<>(body, headers);

            // 2. 调用 Python 服务
            RestTemplate restTemplate = new RestTemplate();
            Map<String, Object> pythonResult = restTemplate.postForObject(
                    PYTHON_SERVICE_URL,
                    requestEntity,
                    Map.class
            );

            // 3. 封装返回
            response.put("code", 200);
            response.put("message", "success");
            response.put("data", pythonResult);

        } catch (Exception e) {
            e.printStackTrace();
            response.put("code", 500);
            response.put("message", "调用 Python 服务失败: " + e.getMessage());
        }

        return response;
    }
}