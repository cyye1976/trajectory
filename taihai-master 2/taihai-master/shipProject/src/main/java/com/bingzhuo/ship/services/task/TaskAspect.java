package com.bingzhuo.ship.services.task;

import kotlin.jvm.Synchronized;
import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.*;
import org.springframework.stereotype.Component;

import javax.annotation.Resource;

@Aspect
@Component
public class TaskAspect {
    @Resource
    private TaskSatatusManager statusManager;

    @Pointcut("@annotation(com.bingzhuo.ship.services.task.annotation.CheckRunning)")
    public void taskPointCut(){
    }

    @Around("taskPointCut()")
    public Object taskAround(ProceedingJoinPoint joinPoint){
        Integer taskId = (Integer)joinPoint.getArgs()[0];
        statusManager.regesitTasking(taskId);
        try {
            return joinPoint.proceed();
        } catch (Throwable throwable) {
            throw new RuntimeException(throwable);
        }finally {
            statusManager.deleteTasking(taskId);
        }
    }
//    @Before("taskPointCut()")
//    public void regesit(JoinPoint joinPoint){
//        Integer taskId = (Integer)joinPoint.getArgs()[0];
//        if (statusManager.isTasking(taskId))
//            throw new RuntimeException("该任务正在处理中");
//        else
//            statusManager.regesitTasking(taskId);
//    }
//
//    @After("taskPointCut()")
//    public void deleteRegesit(JoinPoint joinPoint){
//        Integer taskId = (Integer) joinPoint.getArgs()[0];
//        statusManager.deleteTasking(taskId);
//    }
}
