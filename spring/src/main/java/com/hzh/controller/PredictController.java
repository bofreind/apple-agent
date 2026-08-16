package com.hzh.controller;

import com.hzh.Service.AIService;
import com.hzh.Service.gaoguangpu;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;


import java.util.Map;

@RestController
@RequestMapping("/api")
public class PredictController {

    @Autowired
     gaoguangpu gaoguangpu;

    @PostMapping("/predict")
    public Map<String, Object> predict(@RequestParam("file") MultipartFile file) {
        return gaoguangpu.predict(file);
    }
}
