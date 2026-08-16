package com.hzh.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;

import java.util.HashMap;
import java.util.Map;
import java.util.Base64;

/**
 * 苹果病害识别 + 智能体问答控制器
 * 转发请求到 Python Agent 服务 (http://localhost:8002/chat)
 */
@RestController
@RequestMapping("/api/apple")
public class ApplePestController {

    // Agent 服务地址（支持问答 + 图片识别）
    private static final String AGENT_SERVICE_URL = "http://localhost:8002/chat";

    @PostMapping("/predict")
    public Map<String, Object> predict(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "question", required = false, defaultValue = "请分析这张苹果叶片图片，判断是否有病害。") String question) {

        Map<String, Object> response = new HashMap<>();

        try {
            // 1. 把图片转成 Base64（Agent 服务接收的是 JSON，不是 form-data）
            byte[] fileBytes = file.getBytes();
            String imageBase64 = Base64.getEncoder().encodeToString(fileBytes);

            // 2. 构造 JSON 请求体
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("question", question);
            requestBody.put("image_base64", imageBase64);

            // 3. 设置请求头
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);

            // 4. 调用 Agent 服务
            RestTemplate restTemplate = new RestTemplate();
            Map<String, Object> agentResult = restTemplate.postForObject(
                    AGENT_SERVICE_URL,
                    requestEntity,
                    Map.class
            );

            // 5. 封装返回
            response.put("code", 200);
            response.put("message", "success");
            response.put("data", agentResult);

        } catch (Exception e) {
            e.printStackTrace();
            response.put("code", 500);
            response.put("message", "调用 Agent 服务失败: " + e.getMessage());
        }

        return response;
    }
}