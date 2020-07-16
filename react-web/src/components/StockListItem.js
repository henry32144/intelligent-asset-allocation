import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import IconButton from '@material-ui/core/IconButton';
import DeleteIcon from '@material-ui/icons/Delete';

const useStyles = makeStyles((theme) => ({
}));

export default function StockListItem(props) {
  /* 
    props data structure-
      companyName(string): the name of the company, Ex: Apple Inc.
      companyCode(string): the code of the company in the market, Ex: NASDAQ: AAPL
  */
  const classes = useStyles();

  return (
    <div className={classes.root}>
      <ListItem alignItems="flex-start">
        <ListItemText primary={props.companyName} secondary={props.companyCode} />
        <IconButton edge="end" aria-label="delete">
          <DeleteIcon />
        </IconButton>
      </ListItem>
    </div>
  );
}