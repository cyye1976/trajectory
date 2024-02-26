package com.bingzhuo.ship.entity.vo.restful;

public enum ResponseStatus {

    /**
     * 操作成功
     */
    SUCCESS(2000),
    /**
     * 操作失败
     */
    ERROR(-2000),
    /**
     * 服务器繁忙
     */
    SERVERS_ARE_TOO_BUSY(1),
    /**
     * 未查询到数据
     */
    NOT_FOUND(-1000),
    /**
     * 业务异常
     */
    BIZ_ERROR(10000),
    /**
     * session 失效
     */
    NOT_LOGIN(3),
    /**
     * session 失效
     */
    SESSION_OUT(-100),
    /**
     * 未登陆或者token非法
     */
    INVALID_TOKEN(2001),
    /**
     * 没有权限
     */
    NOT_PERMISSION(2003),
    /**
     * 未知的异常
     */
    UNKOWN_EXCEPTION(3000),
    /**
     * 调用端异常
     */
    CLIENT_EXCEPTION(4000),
    /**
     * 请求参数非法
     */
    PARAM_EXCEPTION(4010),
    /**
     * 服务端异常
     */
    SERVER_EXCEPTION(5000),
    ;

    int code;

    ResponseStatus(int code) {
        this.code = code;
    }
}
