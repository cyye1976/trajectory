package com.bingzhuo.ship.entity.vo.restful;

import org.apache.commons.lang3.builder.ReflectionToStringBuilder;

import java.io.Serializable;

public class Response<T> implements Serializable {

    private static final long serialVersionUID = -5130381241418924208L;

    private int code;       //返回编码
    private String msg;     //消息描述
    private T data;         //返回内容
    private boolean success;//是否操作成功

    private Response(){

    }

    private Response(int code, String msg,boolean success, T data) {
        this.code = code;
        this.msg = msg;
        this.data = data;
        this.success = success;
    }

    public int getCode() {
        return code;
    }

    public String getMsg() {
        return msg;
    }

    public T getData() {
        return data;
    }

    public boolean isSuccess() {
        return success;
    }

    @Override
    public String toString() {
        return ReflectionToStringBuilder.toString(this);
    }

    public static <T> ResponseBuilder builder(){
        return new ResponseBuilder<T>();
    }

    public static class ResponseBuilder<T>{

        private int code;

        private String msg;

        private T data;

        private boolean success;

        public Response build(){
            return new Response<>(this.code,this.msg,this.success,this.data);
        }

        public ResponseBuilder code(ResponseStatus status){
            this.code = status.code;
            return this;
        }

        public ResponseBuilder msg(String msg){
            this.msg = msg;
            return this;
        }

        public ResponseBuilder data(T data){
            this.data = data;
            return this;
        }

        public ResponseBuilder success(boolean success){
            this.success = success;
            return this;
        }
    }

    //构建操作成功返回体
    public static Response success(){
        return builder().success(true).code(ResponseStatus.SUCCESS).build();
    }

    public static Response success(String msg){
        return builder().success(true).code(ResponseStatus.SUCCESS).msg(msg).build();
    }

    public static <T> Response success(String msg, T data){
        return builder().success(true).code(ResponseStatus.SUCCESS).msg(msg).data(data).build();
    }

    //构建没有查询到数据返回体
    public static Response notFound(){
        return builder().success(false).code(ResponseStatus.NOT_FOUND).build();
    }

    public static Response notFound(String msg){
        return builder().success(false).code(ResponseStatus.NOT_FOUND).msg(msg).build();
    }

    //构建访问被拒绝返回体
    public static Response serversAreTooBusy(String msg){
        return builder().success(false).code(ResponseStatus.SERVERS_ARE_TOO_BUSY).msg(msg).build();
    }

    //业务异常
    public static Response bizError(String msg) {
        return builder().success(false).code(ResponseStatus.BIZ_ERROR).code(ResponseStatus.BIZ_ERROR).msg(msg).build();
    }

    //操作失败
    public static Response operationFalse(String msg){
        return builder().success(false).code(ResponseStatus.ERROR).msg(msg).build();
    }

    //传递过来的参数错误返回体
    public static Response paramError(String msg) {
        return builder().success(false).code(ResponseStatus.PARAM_EXCEPTION).msg(msg).build();
    }




}
