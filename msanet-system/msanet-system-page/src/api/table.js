import request from '@/utils/request'

export function getList(params) {
  return request({
    url: '/vue-admin-template/table/list',
    method: 'get',
    params
  })
}

export function httpGet(url){
  return request({
    url: url,
    method: 'get'
  })
}

export function httpPost(url, params){
  return request({
    url: url,
    method: 'post',
    params
  })
}
